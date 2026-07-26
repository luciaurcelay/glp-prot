"""Build the labeled Swiss-Prot eval store by orchestrating InterPLM (RECON §9).

We do NOT reimplement annotation logic — we call InterPLM's own functions:
  extract_annotations.main -> aa_concepts.npz (+ uniprotkb_aa_concepts_columns.txt)
  prepare_eval_set.make_eval_subset -> valid/test concept subsets (shard-range split)
then embed each shard's sequences at the hooked layer with our ActivationExtractor,
producing per-shard activations aligned ROW-FOR-ROW with aa_concepts.npz
(same protein order, CLS/EOS dropped — RECON §9 alignment contract).

Concept columns are read from the materialized column file, never hardcoded `ft_*` (D10).
"""

from __future__ import annotations

import gzip
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# Upstream location of InterPLM's shipped 1k dense-annotated subset (absent from clone).
SUBSET_URLS = [
    "https://raw.githubusercontent.com/ElanaPearl/InterPLM/main/data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz",
    "https://media.githubusercontent.com/media/ElanaPearl/InterPLM/main/data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz",
]


def fetch_uniprot_fasta_slice(dest: Path, n_seqs: int,
                              query: str = "(reviewed:true) AND (length:[80 TO 400])") -> Path:
    """Download up to `n_seqs` Swiss-Prot sequences as FASTA via the UniProt REST API.

    Used for the unlabeled GLP *training* store on the PoC (RECON §10) — a small
    Swiss-Prot slice standing in for UniRef50 (no 12 GB download). Uses the paginated
    `search` endpoint (500/page) and stops once `n_seqs` are collected.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    base = "https://rest.uniprot.org/uniprotkb/search"
    params = {"format": "fasta", "query": query, "size": "500"}
    url = base + "?" + urllib.parse.urlencode(params)
    collected, out = 0, []
    while url and collected < n_seqs:
        req = urllib.request.Request(url, headers={"User-Agent": "glp-plm/0.1"})
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            link = resp.headers.get("Link", "")
        out.append(body)
        collected += body.count(">")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    text = "".join(out)
    # trim to exactly n_seqs records
    records = text.split("\n>")
    if len(records) > n_seqs:
        records = records[:n_seqs]
    text = "\n>".join(records)
    if not text.startswith(">"):
        text = ">" + text.lstrip(">")
    dest.write_text(text)
    print(f"[swissprot] wrote ~{min(collected, n_seqs)} sequences to {dest}")
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
# which is exactly what extract_annotations expects (RECON §9/D10).
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
        acts = acts.detach().cpu().numpy().astype("float32")
        np.save(shard / f"activations_layer{layer}.npy", acts)
        print(f"[swissprot] shard {i}: {acts.shape[0]} aa activations at layer {layer}")


def load_split(processed_dir: Path, split: str, layer: int):
    """Load (activations, labels_df, concept_names) for a split ('valid'/'test').

    labels: dense array (n_aa, n_kept_concepts) from aa_concepts.npz restricted to the
    eval set's kept concepts; concept_names from aa_concepts_columns.txt.
    """
    from scipy import sparse

    split_dir = processed_dir / split
    meta = json.loads((split_dir / "metadata.json").read_text())
    kept = meta["indices_of_concepts_to_keep"]
    concept_names = (split_dir / "aa_concepts_columns.txt").read_text().splitlines()
    shard_ids = meta["shard_source"]

    acts, labels = [], []
    for i in shard_ids:
        shard = processed_dir / f"shard_{i}"
        a = np.load(shard / f"activations_layer{layer}.npy")
        y = sparse.load_npz(shard / "aa_concepts.npz").toarray()[:, kept]
        assert a.shape[0] == y.shape[0], (
            f"shard {i} misalignment: {a.shape[0]} acts vs {y.shape[0]} labels (RECON §9)"
        )
        acts.append(a)
        labels.append(y)
    acts = np.concatenate(acts, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels_df = pd.DataFrame((labels > 0).astype(np.uint8), columns=concept_names)
    return torch.from_numpy(acts), labels_df, concept_names
