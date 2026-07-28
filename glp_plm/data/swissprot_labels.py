"""Build the labeled Swiss-Prot eval store by orchestrating InterPLM (docs/INTERPLM_MAP.md).

We do NOT reimplement annotation logic — we call InterPLM's own functions:
  extract_annotations.main -> aa_concepts.npz (+ uniprotkb_aa_concepts_columns.txt)
  prepare_eval_set.make_eval_subset -> valid/test concept subsets (shard-range split)
then embed each shard's sequences at the hooked layer with our ActivationExtractor,
producing per-shard activations aligned ROW-FOR-ROW with aa_concepts.npz
(same protein order, CLS/EOS dropped — the alignment contract; see docs/INTERPLM_MAP.md).

Concept columns are read from the materialized column file; never hardcode `ft_*` — they
are UniProt display-name columns (see docs/INTERPLM_MAP.md).
"""

from __future__ import annotations

import gzip
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Upstream location of InterPLM's shipped 1k dense-annotated subset (absent from clone).
SUBSET_URLS = [
    "https://raw.githubusercontent.com/ElanaPearl/InterPLM/main/data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz",
    "https://media.githubusercontent.com/media/ElanaPearl/InterPLM/main/data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz",
]


def read_accessions(tsv: Path) -> set:
    """Accessions in a UniProt TSV (the `Entry` column) — used to exclude eval proteins."""
    import csv
    with open(tsv) as f:
        r = csv.DictReader(f, delimiter="\t")
        key = next(c for c in r.fieldnames if c.lower() in ("entry", "accession"))
        return {row[key] for row in r if row.get(key)}


def fetch_uniprot_fasta_slice(dest: Path, n_seqs: int,
                              query: str = "(reviewed:true) AND (length:[80 TO 400])",
                              exclude: Optional[set] = None) -> Path:
    """Download up to `n_seqs` Swiss-Prot sequences as FASTA via the UniProt REST API.

    Used for the unlabeled GLP *training* store — a small Swiss-Prot slice standing in
    for UniRef50 (no 12 GB download). Paginated `search` endpoint, 500/page.

    `exclude` drops accessions (e.g. every protein in the labeled eval set). Without it
    the training and eval queries overlap almost completely — measured 98.4% — which
    hands GLP activations it will later be probed on, an advantage the SAE (trained on
    UniRef50) does not get. Always exclude the eval set.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    exclude = exclude or set()
    base = "https://rest.uniprot.org/uniprotkb/search"
    params = {"format": "fasta", "query": query, "size": "500"}
    url = base + "?" + urllib.parse.urlencode(params)
    kept, skipped = [], 0
    while url and len(kept) < n_seqs:
        req = urllib.request.Request(url, headers={"User-Agent": "glp-plm/0.1"})
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            link = resp.headers.get("Link", "")
        for rec in body.split("\n>"):
            rec = rec.lstrip(">")
            if not rec.strip():
                continue
            m = re.match(r"\w+\|([^|]+)\|", rec)
            if m and m.group(1) in exclude:
                skipped += 1
                continue
            kept.append(">" + rec.rstrip("\n"))
            if len(kept) >= n_seqs:
                break
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    dest.write_text("\n".join(kept) + "\n")
    print(f"[swissprot] wrote {len(kept)} sequences to {dest}"
          + (f" (excluded {skipped} eval proteins)" if exclude else ""))
    return dest


def read_fasta(path: Path) -> List[str]:
    """Return the list of sequences from a FASTA file."""
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


# UniProt REST field IDs matching InterPLM's Option B query (README:163). The TSV
# header uses display names ("Sequence", "Length", "AlphaFoldDB", "Active site", ...)
# which is exactly what extract_annotations expects (docs/INTERPLM_MAP.md).
_ANNOT_FIELDS = ",".join([
    "accession", "reviewed", "protein_name", "length", "sequence", "ec",
    "ft_act_site", "ft_binding", "cc_cofactor", "ft_disulfid", "ft_carbohyd",
    "ft_lipid", "ft_mod_res", "ft_signal", "ft_transit", "ft_helix", "ft_turn",
    "ft_strand", "ft_coiled", "cc_domain", "ft_compbias", "ft_domain", "ft_motif",
    "ft_region", "ft_zn_fing", "xref_alphafolddb",
])


def fetch_uniprot_annotated_tsv(
    dest: Path,
    n_prot: int,
    query: str = "(reviewed:true) AND (annotation_score:5) AND (length:[50 TO 400])",
) -> Path:
    """Download up to `n_prot` densely-annotated Swiss-Prot proteins as a UniProt TSV.

    This is the eval-set source for the PoC. The shipped 1k dense-annotated subset is
    NOT committed to the InterPLM repo (only referenced in its README), so we pull a
    fresh slice via the same REST query family InterPLM documents (README Option B).
    Output is consumable directly by interplm.analysis.concepts.extract_annotations.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    base = "https://rest.uniprot.org/uniprotkb/search"
    params = {"format": "tsv", "fields": _ANNOT_FIELDS, "query": query, "size": "500"}
    url = base + "?" + urllib.parse.urlencode(params)
    header, rows = None, []
    while url and len(rows) < n_prot:
        req = urllib.request.Request(url, headers={"User-Agent": "glp-plm/0.1"})
        with urllib.request.urlopen(req) as resp:
            page = resp.read().decode("utf-8").splitlines()
            link = resp.headers.get("Link", "")
        if not page:
            break
        if header is None:
            header = page[0]
        rows.extend(page[1:])
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    rows = rows[:n_prot]
    dest.write_text("\n".join([header] + rows) + "\n")
    print(f"[swissprot] wrote {len(rows)} annotated proteins to {dest}")
    return dest


def fetch_1k_subset(dest: Path) -> Path:
    """Download the 1k dense-annotated Swiss-Prot subset if not already present."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    last_err = None
    for url in SUBSET_URLS:
        try:
            print(f"[swissprot] fetching {url}")
            urllib.request.urlretrieve(url, dest)
            # sanity: must be gzip
            with gzip.open(dest, "rb") as f:
                f.read(64)
            return dest
        except Exception as e:  # try next mirror
            last_err = e
            if dest.exists():
                dest.unlink()
    raise RuntimeError(
        f"Could not download the 1k subset from any mirror ({last_err}). "
        f"Download it manually to {dest} (from the InterPLM GitHub repo)."
    )


def _patch_interplm_sharding() -> None:
    """Compat shim: InterPLM's shard_protein_data uses np.array_split(df, n), which
    returns ndarrays (not DataFrames) under our pandas, breaking `.reset_index`.
    Replace it with an index-split version. Contained here; the submodule is untouched.
    """
    import numpy as np
    from interplm.analysis.concepts import extract_annotations as ea

    def shard_protein_data(df, output_dir, n_shards):
        np.random.seed(42)
        for shard_id, idx in enumerate(np.array_split(np.arange(len(df)), n_shards)):
            df_shard = df.iloc[idx].reset_index(drop=True)
            shard_dir = output_dir / f"shard_{shard_id}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            df_shard.to_csv(shard_dir / "protein_data.tsv", sep="\t", index=False)

    ea.shard_protein_data = shard_protein_data


def build_annotations(subset_tsv_gz: Path, processed_dir: Path,
                      n_shards: int, min_required_instances: int = 10) -> None:
    """Run InterPLM's extract_annotations to materialize per-AA concept labels."""
    _patch_interplm_sharding()
    from interplm.analysis.concepts.extract_annotations import main as extract_main

    extract_main(
        input_uniprot_path=subset_tsv_gz,
        output_dir=processed_dir,
        n_shards=n_shards,
        min_required_instances=min_required_instances,
    )


def build_eval_subsets(processed_dir: Path, valid_range: Tuple[int, int],
                       test_range: Tuple[int, int], min_aa: int, min_domains: int) -> None:
    """Run InterPLM's prepare_eval_set to create valid/test concept subsets."""
    from interplm.analysis.concepts.prepare_eval_set import make_eval_subset

    make_eval_subset(
        uniprot_dir=processed_dir,
        valid_shard_range=valid_range,
        test_shard_range=test_range,
        min_aa_per_concept=min_aa,
        min_domains_per_concept=min_domains,
    )


def embed_shards(processed_dir: Path, shard_ids: List[int], model_name: str,
                 layer: int, device: str | None = None) -> None:
    """Embed each shard's sequences at `layer`, aligned to aa_concepts.npz rows.

    Saves `processed_dir/shard_{i}/activations_layer{layer}.npy` of shape (n_aa, d).
    """
    from glp_plm.data.activation_extractor import ActivationExtractor

    extractor = ActivationExtractor(model_name=model_name, layer=layer, device=device)
    for i in shard_ids:
        shard = processed_dir / f"shard_{i}"
        df = pd.read_csv(shard / "protein_data.tsv", sep="\t")
        seq_col = next(c for c in df.columns if c.lower() == "sequence")
        seqs = [str(s) for s in df[seq_col].tolist()]
        acts = extractor.extract_many(seqs, batch_size=8)      # (sum_len, d), cls/eos dropped
        # fp16 to match the training store; float32 doubles the eval store for no
        # benefit (8 -> 16 GB at InterPLM's full 50k proteins). Consumers cast up.
        acts = acts.detach().cpu().numpy().astype("float16")
        np.save(shard / f"activations_layer{layer}.npy", acts)
        print(f"[swissprot] shard {i}: {acts.shape[0]} aa activations at layer {layer}")


def split_shards(processed_dir: Path, split: str, n_shards: Optional[int] = None) -> List[int]:
    """Shard ids backing a split, optionally truncated to the first `n_shards`."""
    meta = json.loads((Path(processed_dir) / split / "metadata.json").read_text())
    ids = list(meta["shard_source"])
    return ids[:n_shards] if n_shards else ids


def load_split(processed_dir: Path, split: str, layer: int,
               n_shards: Optional[int] = None, binarize: bool = True):
    """Load (activations, labels_df, concept_names) for a split ('valid'/'test').

    labels: dense array (n_aa, n_kept_concepts) from aa_concepts.npz restricted to the
    eval set's kept concepts; concept_names from aa_concepts_columns.txt.

    `binarize=True` gives 0/1 (what AUC probing wants). `binarize=False` keeps the
    integer **domain-instance index** each residue belongs to, which the F1 protocol
    needs for recall-per-domain (InterPLM counts distinct domains touched).
    """
    from scipy import sparse

    split_dir = processed_dir / split
    meta = json.loads((split_dir / "metadata.json").read_text())
    kept = meta["indices_of_concepts_to_keep"]
    concept_names = (split_dir / "aa_concepts_columns.txt").read_text().splitlines()
    shard_ids = list(meta["shard_source"])
    if n_shards:                     # bound the split (keeps raw_mlp caches sane)
        shard_ids = shard_ids[:n_shards]

    acts, labels = [], []
    for i in shard_ids:
        shard = processed_dir / f"shard_{i}"
        a = np.load(shard / f"activations_layer{layer}.npy")
        y = sparse.load_npz(shard / "aa_concepts.npz").toarray()[:, kept]
        assert a.shape[0] == y.shape[0], (
            f"shard {i} misalignment: {a.shape[0]} acts vs {y.shape[0]} labels (docs/INTERPLM_MAP.md)"
        )
        acts.append(a)
        labels.append(y)
    acts = np.concatenate(acts, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels_df = pd.DataFrame((labels > 0).astype(np.uint8) if binarize else labels,
                             columns=concept_names)
    return torch.from_numpy(acts), labels_df, concept_names
