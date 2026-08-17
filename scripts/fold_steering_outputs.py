#!/usr/bin/env python
"""Fold a steering run's generated sequences with ESMFold2 -> pLDDT / pTM sidecar CSV.

Why this is a SEPARATE stage rather than a call inside run_steering.py's generation loop:

  1. Different interpreter. ESMFold2 needs the `esm` package and lives in its own Python
     3.12 venv on MN5 (`containers/esmfold2/venv312`); the steering pipeline runs on 3.11
     and has no `esm` at all. Nothing can import both.
  2. Restartability. Folding is the expensive part of the pipeline per sequence, so it has
     to be resumable independently of generation — an interrupted fold pass should cost one
     sequence, not the whole run (same principle as run_probing.py's checkpointing).

So this reads a finished run's CSV, writes `fold_confidence.csv` beside it keyed on
sample_id, and never touches the generation output. Deliberately depends on stdlib + torch
+ transformers ONLY (no glp_plm / _bootstrap import), because venv312 is not the project
venv and does not have the project's dependencies.

Why folding at all: pseudo-perplexity, the naturalness axis Exp B has been using, is
actively misleading here — the GLP correction's Leu/Val enrichment makes sequences *easier*
to predict (pppl 15.0 -> 13.9) while pushing composition away from natural. Folding
confidence is the field-standard, independent axis: it asks whether the sequence specifies a
structure at all, which Leu/Val stuffing cannot fake (hydrophobic-enriched sequences are
aggregation-prone, not well-folded).

Scales (measured, not assumed — see scripts/cluster/submit_fold.sh's notes):
  * ESMFold2 returns pLDDT in [0,1] (natural ubiquitin ~0.80). Recorded here x100, i.e. the
    conventional 0-100 pLDDT everyone reports, matching the existing `plddt` column.
  * pTM / ipTM are TM-score units in [0,1] and recorded as-is.
`num_sampling_steps` is recorded per row: this is a diffusion model, so a confidence number
is only comparable against others at the same step count (same discipline as quoting FD with
its sampler step count).

Usage (on MN5, via the launcher which sets up the 3.12 module stack):
  sbatch scripts/cluster/submit_fold.sh <steering-out-dir> [extra args...]
Directly:
  <venv312>/bin/python scripts/fold_steering_outputs.py --run-dir <steering-out-dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

CSV_CANDIDATES = ("expB_diffmean_gravy.csv", "expA_sae_zincfinger.csv")
SIDECAR = "fold_confidence.csv"
FIELDS = ["sample_id", "direction", "alpha", "corrected", "length", "plddt", "plddt_ca",
          "ptm", "iptm", "pae_mean", "num_sampling_steps", "num_loops", "model_id",
          "seq_sha1", "fold_error"]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def find_input(run_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"--input {p} does not exist")
        return p
    for name in CSV_CANDIDATES:
        if (run_dir / name).exists():
            return run_dir / name
    sys.exit(f"no {' or '.join(CSV_CANDIDATES)} found in {run_dir} — pass --input explicitly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="steering out-dir holding the run's CSV")
    ap.add_argument("--input", help="override the generated-sequences CSV path")
    ap.add_argument("--out", help=f"override the sidecar path (default <run-dir>/{SIDECAR})")
    ap.add_argument("--model-id", default="biohub/ESMFold2",
                    help="only biohub/ESMFold2 is prefetched in the shared HF cache; "
                         "biohub/ESMFold2-Fast would need a download and MN5 is offline")
    ap.add_argument("--num-sampling-steps", type=int, default=20,
                    help="diffusion steps. Default 20 rather than the README's 50: measured "
                         "on this run's own sequences the two agree to ~0.01 pLDDT and "
                         "~0.003 pTM (far below the between-condition differences) at half "
                         "the cost (~2.3s vs ~4.4s per 145-residue sequence on an H100)")
    ap.add_argument("--num-loops", type=int, default=3)
    ap.add_argument("--limit-per-cell", type=int,
                    help="fold at most N sequences per (direction, alpha, corrected) cell — "
                         "for a cheap smoke pass. Default: fold everything (a 300-sequence "
                         "Exp B run is only ~12 min at 20 steps)")
    ap.add_argument("--restart", action="store_true",
                    help="ignore an existing sidecar instead of resuming from it")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    in_path = find_input(run_dir, args.input)
    out_path = Path(args.out) if args.out else run_dir / SIDECAR
    rows = list(csv.DictReader(open(in_path)))
    if not rows:
        sys.exit(f"{in_path} has no rows")
    print(f"[fold] input {in_path} ({len(rows)} rows)", flush=True)

    done: set[str] = set()
    if out_path.exists() and not args.restart:
        with open(out_path) as f:
            for r in csv.DictReader(f):
                # a row that errored is retried on the next pass rather than being cached
                if not r.get("fold_error"):
                    done.add(r["sample_id"])
        print(f"[fold] resuming: {len(done)} already folded in {out_path}", flush=True)

    # cell-based subsampling operates on the INPUT order so it is deterministic
    todo = []
    per_cell: dict[tuple, int] = {}
    for r in rows:
        if r["sample_id"] in done:
            continue
        if not r.get("sequence"):
            continue
        cell = (r.get("direction") or "up", r["alpha"], r["corrected"])
        if args.limit_per_cell is not None:
            if per_cell.get(cell, 0) >= args.limit_per_cell:
                continue
            per_cell[cell] = per_cell.get(cell, 0) + 1
        todo.append(r)
    print(f"[fold] {len(todo)} sequences to fold "
          f"(steps={args.num_sampling_steps}, loops={args.num_loops})", flush=True)
    if not todo:
        print("[fold] nothing to do")
        return

    import torch
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    print(f"[fold] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    t0 = time.time()
    model = ESMFold2Model.from_pretrained(args.model_id)
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    load_s = time.time() - t0
    print(f"[fold] loaded {args.model_id} in {load_s:.1f}s", flush=True)
    # rate/ETA must exclude the one-off model load, or the per-sequence figure is inflated
    # (measured: a 20-sequence pass reported 5.7s/seq when folding actually cost ~2.6s/seq,
    # because the ~61s load was being amortised into it — enough to mis-size a full run 2x)
    t_fold = time.time()

    def scalar(out, key, scale=1.0):
        """Mean of `key` if present, else None. ESMFold2 returns tensors of varying rank
        (per-atom / per-residue / scalar), so everything is reduced with .mean(). Keyed with
        [] rather than .get so it works whether the model returns a dict or a ModelOutput."""
        try:
            v = out[key]
        except (KeyError, TypeError, IndexError):
            return None
        if v is None:
            return None
        try:
            return float(v.float().mean()) * scale
        except Exception:
            return None

    mode = "a" if (out_path.exists() and not args.restart and done) else "w"
    n_ok = n_err = 0
    with open(out_path, mode, newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=FIELDS)
        if mode == "w":
            w.writeheader()
        for i, r in enumerate(todo, 1):
            seq = r["sequence"]
            rec = {"sample_id": r["sample_id"], "direction": r.get("direction") or "up",
                   "alpha": r["alpha"], "corrected": r["corrected"], "length": len(seq),
                   "num_sampling_steps": args.num_sampling_steps,
                   "num_loops": args.num_loops, "model_id": args.model_id,
                   "seq_sha1": sha1(seq), "fold_error": ""}
            try:
                with torch.no_grad():
                    out = model.infer_protein(seq, num_loops=args.num_loops,
                                              num_sampling_steps=args.num_sampling_steps)
                # pLDDT x100 -> conventional 0-100; pTM/ipTM already TM-score units
                rec.update(plddt=scalar(out, "plddt", 100.0),
                           plddt_ca=scalar(out, "plddt_ca", 100.0),
                           ptm=scalar(out, "ptm"), iptm=scalar(out, "iptm"),
                           pae_mean=scalar(out, "pae"))
                n_ok += 1
            except Exception as e:  # keep going; the row is retried on the next pass
                rec["fold_error"] = f"{type(e).__name__}: {e}"[:200]
                n_err += 1
                print(f"[fold] WARNING sample_id={rec['sample_id']} failed: "
                      f"{rec['fold_error']}", flush=True)
            w.writerow(rec)
            fout.flush()   # flush per row so an interrupted job resumes cleanly
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t_fold
                print(f"[fold] {i}/{len(todo)}  ok={n_ok} err={n_err}  {el/i:.1f}s/seq "
                      f"(+{load_s:.0f}s load)  eta={(len(todo)-i)*el/i/60:.1f}min", flush=True)

    print(f"[fold] wrote {out_path} (ok={n_ok}, errors={n_err})")
    if n_err:
        print(f"[fold] {n_err} row(s) errored and were NOT cached — rerun to retry them")


if __name__ == "__main__":
    main()
