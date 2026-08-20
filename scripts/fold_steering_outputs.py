#!/usr/bin/env python
"""Fold a steering run's generated sequences with ESMFold2 -> pLDDT / pTM / DSSP sidecar CSV.

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

Also runs the real DSSP binary (module dssp/4.5.3, /apps/ACC/DSSP/4.5.3/GCC/bin/mkdssp --
confirmed present on MN5, not the pydssp approximation also installed in this venv) on
each fold's own PDB (`model.output_to_pdb(out)` -- reuses the same forward pass, no extra
inference) to get per-residue secondary structure. Exp A's zinc-finger concept score is a
sequence motif (C2H2 regex); no such sequence-only oracle exists for a structural concept
like "Coiled coil" (see run_steering.py's `--concept-detector` docstring), and a coiled
coil IS a structural feature (a continuous, extended alpha helix), so its ground-truth
check has to look at the folded backbone, not the sequence. `helix_frac`/`max_helix_run`
are computed from DSSP's H/G/I classes (E/B for strand_frac/max_strand_run); `ss_hit`
(--target-ss's frac >= threshold, default
0.7) is a single-chain proxy -- a real coiled coil is defined by inter-helix packing
(register, heptad a/d core) between >=2 chains, which a monomer fold cannot show. What
this DOES show: whether steering produced a sequence that folds into one long helix, the
necessary (not sufficient) structural precondition. State it as that, not as "forms a
coiled coil".

Usage (on MN5, via the launcher which sets up the 3.12 module stack):
  sbatch scripts/cluster/submit_fold.sh <steering-out-dir> [extra args...]
Directly:
  <venv312>/bin/python scripts/fold_steering_outputs.py --run-dir <steering-out-dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_DSSP_BIN = "/apps/ACC/DSSP/4.5.3/GCC/bin/mkdssp"
HELIX_CODES = {"H", "G", "I"}  # alpha / 3-10 / pi helix -- DSSP's helical classes
STRAND_CODES = {"E", "B"}      # extended beta strand / isolated beta bridge


def run_dssp(pdb_text: str, dssp_bin: str) -> str:
    """mkdssp needs real files, not stdin/stdout -- round-trip through a temp dir.
    Returns the per-residue SS string (one char/residue, chain-break '!' rows dropped,
    ' ' meaning DSSP's own coil/loop call, in original PDB order)."""
    with tempfile.TemporaryDirectory() as td:
        pdb_path = Path(td) / "model.pdb"
        dssp_path = Path(td) / "model.dssp"
        pdb_path.write_text(pdb_text)
        subprocess.run([dssp_bin, "--output-format", "dssp", str(pdb_path), str(dssp_path)],
                       check=True, capture_output=True, text=True, timeout=60)
        lines = dssp_path.read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("  #  RESIDUE")), None)
    if start is None:
        raise RuntimeError("mkdssp output has no '  #  RESIDUE' header — unexpected format")
    ss = []
    for l in lines[start + 1:]:
        if len(l) < 17 or l[13] == "!":  # chain break placeholder row
            continue
        ss.append(l[16] if l[16] != " " else "-")
    return "".join(ss)


def _max_run(ss: str, codes: set) -> int:
    best = cur = 0
    for c in ss:
        cur = cur + 1 if c in codes else 0
        best = max(best, cur)
    return best


def dssp_summary(ss: str) -> dict:
    if not ss:
        return {"helix_frac": None, "max_helix_run": 0, "strand_frac": None, "max_strand_run": 0}
    return {"helix_frac": sum(c in HELIX_CODES for c in ss) / len(ss),
            "max_helix_run": _max_run(ss, HELIX_CODES),
            "strand_frac": sum(c in STRAND_CODES for c in ss) / len(ss),
            "max_strand_run": _max_run(ss, STRAND_CODES)}

# Any steered property: expB_diffmean_{gravy,charge,instability}.csv. Globbed
# rather than imported from glp_plm.eval.steering_properties because this stage
# runs in the separate ESMFold2 venv, which has no glp_plm.
CSV_GLOB = "expB_diffmean_*.csv"
CSV_CANDIDATES = ("expA_sae_zincfinger.csv",)
SIDECAR = "fold_confidence.csv"
FIELDS = ["sample_id", "direction", "alpha", "corrected", "length", "plddt", "plddt_ca",
          "ptm", "iptm", "pae_mean", "ss_string", "helix_frac", "max_helix_run",
          "strand_frac", "max_strand_run", "ss_hit", "num_sampling_steps", "num_loops",
          "model_id", "seq_sha1", "fold_error", "dssp_error"]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def find_input(run_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"--input {p} does not exist")
        return p
    hits = sorted(run_dir.glob(CSV_GLOB))
    if len(hits) > 1:
        sys.exit(f"{run_dir} holds {len(hits)} Exp B CSVs ({[h.name for h in hits]}) — "
                 "a run steers one property; pass --input explicitly")
    if hits:
        return hits[0]
    for name in CSV_CANDIDATES:
        if (run_dir / name).exists():
            return run_dir / name
    sys.exit(f"no {CSV_GLOB} or {' or '.join(CSV_CANDIDATES)} found in {run_dir} — "
             "pass --input explicitly")


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
    ap.add_argument("--dssp-bin", default=DEFAULT_DSSP_BIN,
                    help="mkdssp binary (module dssp/4.5.3 on MN5); pass '' to skip DSSP "
                         "entirely and only get plddt/ptm as before")
    ap.add_argument("--target-ss", choices=["helix", "strand"], default="helix",
                    help="which DSSP class 'ss_hit' reflects: 'helix' (H/G/I -- Coiled coil "
                         "and Helix both want this) or 'strand' (E/B -- Beta strand). "
                         "helix_frac/strand_frac/max_*_run are always both recorded "
                         "regardless of this choice.")
    ap.add_argument("--ss-frac-threshold", type=float, default=None,
                    help="ss_hit = (helix_frac or strand_frac, per --target-ss) >= this. "
                         "Default is per-target: 0.7 for helix (a long helix can span most "
                         "of a short chain, so the naive whole-chain threshold works), 0.10 "
                         "for strand -- calibrated against 300 real Swiss-Prot Beta strand "
                         "windows (scripts/calibrate_beta_strand.py, 2026-08-20): 0.7 never "
                         "fires on a fragment this short (real strand content comes in short "
                         "runs, not chain-spanning), and 0.10 beat every hairpin-run variant "
                         "tried, including a hand-picked >=2-runs/<=6-residue-gap rule (test "
                         "F1 0.264 vs 0.175) -- see that script's docstring for the full "
                         "candidate table.")
    ap.add_argument("--save-pdb-dir",
                    help="write each fold's PDB to <dir>/<sample_id>_a<alpha>_<naive|corr>.pdb. "
                         "Off by default (PDBs are large; every prior fold pass here discarded "
                         "them after DSSP). ESMFold has to be RE-RUN to get a PDB for a row "
                         "that was already folded without this flag -- there's no cached "
                         "structure to reuse -- so pair this with --restart when adding PDBs "
                         "to an existing fold_confidence.csv.")
    args = ap.parse_args()
    if args.ss_frac_threshold is None:
        args.ss_frac_threshold = 0.7 if args.target_ss == "helix" else 0.10
    if args.save_pdb_dir:
        args.save_pdb_dir = Path(args.save_pdb_dir)
        args.save_pdb_dir.mkdir(parents=True, exist_ok=True)

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
                   "seq_sha1": sha1(seq), "fold_error": "", "dssp_error": ""}
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
                pdb_text = None
                if args.dssp_bin or args.save_pdb_dir:
                    pdb_text = model.output_to_pdb(out)
                if args.save_pdb_dir:
                    pdb_name = (f"{rec['sample_id']}_a{rec['alpha']}_"
                               f"{'corr' if rec['corrected'] in ('True', True) else 'naive'}.pdb")
                    (args.save_pdb_dir / pdb_name).write_text(pdb_text)
                if args.dssp_bin:
                    try:
                        ss = run_dssp(pdb_text, args.dssp_bin)
                        summ = dssp_summary(ss)
                        target_frac = summ["helix_frac"] if args.target_ss == "helix" else summ["strand_frac"]
                        rec.update(ss_string=ss, helix_frac=summ["helix_frac"],
                                   max_helix_run=summ["max_helix_run"],
                                   strand_frac=summ["strand_frac"],
                                   max_strand_run=summ["max_strand_run"],
                                   ss_hit=(target_frac is not None and
                                           target_frac >= args.ss_frac_threshold))
                    except Exception as e:
                        rec["dssp_error"] = f"{type(e).__name__}: {e}"[:200]
                        print(f"[fold] WARNING dssp sample_id={rec['sample_id']} failed: "
                              f"{rec['dssp_error']}", flush=True)
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
