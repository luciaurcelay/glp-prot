# INTERPLM_MAP — reuse map for the InterPLM & GLP clones

Where to find (and what to reuse from) the two submodules, plus the live gotchas.
`docs/PLAN.md` is the spec; this file is the "don't reimplement it, call this" index.
Paths are relative to the repo root. Verified against the clones during W1.

## InterPLM (`InterPLM/interplm/…`)
| Need | Reuse | Location |
|---|---|---|
| Embed a sequence at a layer | `embedders.esm.ESM` / `embedders.get_embedder("esm", model_name=…)`; `embed_single_sequence(seq, layer)`, `extract_embeddings_multiple_layers(seqs, layers, …)` | `embedders/esm.py:116,242` |
| Load a released SAE | `sae.inference.load_sae_from_hf(plm_model, plm_layer)` → `ae_normalized.pt` | `sae/inference.py:66` (HF layer sets `:24-27`) |
| Encode features (probing/F1) | `sae.dictionary.*.encode_feat_subset(x, feat_list, normalize_features=True)` | `sae/dictionary.py:233` |
| SAE steering (recon + error, NNsight) | `sae.intervention.get_esm_output_with_intervention(...)` | `sae/intervention.py` |
| Swiss-Prot per-AA labels | `analysis.concepts.extract_annotations.main(...)` → `aa_concepts.npz` + `uniprotkb_aa_concepts_columns.txt` | `analysis/concepts/extract_annotations.py` |
| Val/test concept subsets | `analysis.concepts.prepare_eval_set.make_eval_subset(...)` (defaults `min_aa=1500`, `min_domains=25`) | `analysis/concepts/prepare_eval_set.py:15-22` |
| Domain-adjusted F1 | `analysis.concepts.calculate_f1.calculate_f1`, `compare_activations.calc_metrics_sparse`; `concept_constants.is_aa_level_concept` picks AA- vs domain-level recall | `analysis/concepts/calculate_f1.py:64-86`, `compare_activations.py:106-136` |
| % Loss Recovered | `train.fidelity.calculate_loss_recovered(...)` (full-sequence CE with activation patching, **not** 15% MLM) | `train/fidelity.py:49` |

## GLP reference (`generative_latent_prior/glp/…`)
| Need | Reuse / reference | Location |
|---|---|---|
| Denoiser architecture | `TransformerMLPBlock` / `TransformerMLPDenoiser` (LayerNorm; time-modulated gate) | `denoiser.py:94-211` |
| Flow-matching objective | `fm_prepare` — `noisy=(1-σ)z0+σε`, `target=ε-z0` (σ ≡ our `t`) | `flow_matching.py:32-35` |
| Meta-neuron for probing | hooks **`denoiser.model.layers.{i}.down_proj` input**; probing protocol (top-512 → 1-D `LogisticRegressionCV`) | `script_probe.py:128,28-53` (`u=0.9` default `:211`) |
| SDEdit steering cleanup | `postprocess_on_manifold` (`u=0.5`, 20 steps) | `script_steer.py:11-45` |
| Width rule / hparams | `d_model=2·d_input`, `d_mlp=4·d_input`; lr 5e-5, batch 4096, bf16 | `configs/train_llama8b_static.yaml:22-45` |
| Normalizer format | stores `{mean, var}`; `(x-mean)/√var` | `denoiser.py:19-71` |

## Live gotchas (the expensive ones)
- **Hook parity.** Train GLP on the **raw** residual stream after ESM-2-8M block 4 =
  `hidden_states[4]`, cls/eos dropped (the embedder already drops them — don't re-strip).
  The SAE normalizes its input internally (√d), so feed GLP the raw activation and do
  GLP's own mean/var standardization separately. Guarded by `tests/test_hook_parity.py`.
- **Layer indexing.** InterPLM "layer L" = `hidden_states[L]` = output of block L
  (`esm.py:179`, `intervention.py:52-53`); `hidden_states[0]` = embeddings.
- **Meta-neuron = down_proj input**, `SiLU(gate_proj(x)·time_proj(t_emb))·up_proj(x)` —
  post-gating and **includes the time factor** (function of probing `u`).
- **Block uses LayerNorm** (not RMSNorm); time conditioning is **plain multiplicative** on
  the gate (no `(1+γ)`/`β`, up path unmodulated). `denoiser.py:111,119-124`.
- **Concept filter** defaults: `>25 domains OR >1,500 AAs` (`prepare_eval_set.py:21-22`).
  `min_required_instances` (README's "10") is a different, subcategory-inclusion knob.
- **Concept names** are UniProt display columns + subcategory expansions (read
  `uniprotkb_aa_concepts_columns.txt`) — the `ft_*` are REST download field IDs, not column names.
- **Eval proteins** are pre-filtered to those with an AlphaFoldDB structure, length ≤ 1022
  (`extract_annotations.py:151,256`) — so the biophysical candidate signal is cheap.
- **The 1k dense-annotated subset** referenced in InterPLM's README is **not committed**
  upstream — use a UniProt REST slice instead (`glp_plm/data/swissprot_labels.py`).
- **F1 thresholds don't port to dense features.** InterPLM's sweep is a *fraction of each
  feature's max activation*, which assumes sparse non-negative ReLU features. On dense
  signed activations it degenerates (56% of residues fire at thr=0, 0.07% at thr=0.5) and
  one-sided `f > thr` misses negatively-encoded concepts. We use rate-matched, two-sided
  operating points instead (`glp_plm/eval/probing_f1.py`); the F1 definitions are unchanged.
- **F1 is prevalence-sensitive** — evaluate at natural prevalence on the full split, never
  on the balanced subsample the AUC path uses.

## Environment / integration gotchas (see `SETUP.md`)
- **Do not `pip install -e InterPLM`.** Its `setup.py` misses `interplm/train/__init__.py`
  (non-editable drops `interplm.train`), and Homebrew Python's `sitecustomize` shadows venv
  site hooks. Put both on `sys.path` via `scripts/_bootstrap.py` / `tests/conftest.py`.
- **pandas compat:** InterPLM's `shard_protein_data` uses `np.array_split(df, n)` which
  returns ndarrays under our pandas; a contained shim in `glp_plm/data/swissprot_labels.py`
  (`_patch_interplm_sharding`) fixes it without editing the submodule.
- **Canonical model** `facebook/esm2_t6_8M_UR50D` (EsmModel/EsmForMaskedLM) — never the
  NVIDIA `nv_esm` repackaging (would break hook parity).
