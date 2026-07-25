# RECON — verifying PLAN.md ⚠/provisional items against the clones

Session scope: **A. RECON** (this file), **B. SCAFFOLD**, **C. TESTS**. No training, no bulk download.
All citations are to files in the two submodules:
- `InterPLM/` = clone of `ElanaPearl/InterPLM`
- `generative_latent_prior/` = clone of `g-luo/generative_latent_prior` (the GLP reference implementation — this is the actual released code, so it is ground truth over the paper text for anything implementation-level).

Legend: ✅ confirmed as planned · ⚠️ **DELTA** (plan is wrong/imprecise — fix proposed) · ℹ️ note.

---

## 1. Embedding API (PLAN §2.2, §5, §6.1, D2, hook-parity test)

⚠️ **DELTA D1 — the API name and signature in the plan do not exist.**
The plan repeatedly cites `interplm.esm.embed.embed_single_sequence(sequence, model_name, layer)`. There is **no `interplm.esm` package**. The real embedder is a class:

- `interplm/embedders/esm.py:20` → `class ESM(BaseEmbedder)`
- Constructor takes `model_name` (default `"facebook/esm2_t6_8M_UR50D"`), `device`, `max_length=1024` (`esm.py:43-48`).
- Factory: `interplm/embedders/__init__.py:8` → `get_embedder("esm", model_name=...)`.
- Single-sequence helper: `ESM.embed_single_sequence(self, sequence, layer)` (`esm.py:242`) — **signature is `(sequence, layer)`**; `model_name` is a constructor arg, not a call arg. Returns `np.ndarray` `(seq_len, d)`.
- Bulk helper for the training store: `ESM.extract_embeddings_multiple_layers(sequences, layers, batch_size, shuffle)` (`esm.py:116`) → dict `{layer: (total_tokens, d)}`.

**Fix:** `activation_extractor.py` wraps `get_embedder("esm", model_name=cfg.model_name)` and calls `embed_single_sequence(seq, layer)` / `extract_embeddings_multiple_layers(...)`. Update PLAN §2.2, §5, §6.1, D2 to the real names. (Scaffold already codes against the real API and gracefully degrades to a documented fallback when InterPLM/torch aren't importable, so the hook-parity test can run once deps are installed.)

⚠️ **DELTA D2 — CLS/EOS are already dropped inside the embedder.**
`extract_embeddings_multiple_layers` slices `layer_output[seq_idx, 1:seq_len+1, :]` per sequence (`esm.py:181-186`), i.e. it removes `<cls>` (pos 0) and `<eos>`/pad and returns a **flat `(n_tokens, d)`** already stripped. `embed_single_sequence` returns `(seq_len, d)` likewise stripped.

**Consequence:** the plan's "wrapper drops cls/eos/pad" would **double-drop**. Our wrapper must NOT strip again. The hook-parity test compares our extractor **exactly** to `embed_single_sequence` output (the "up to the cls/eos drop" caveat in §5 is moot — parity is exact, because both already exclude the special tokens). Fix PLAN §5/§6.1 wording.

ℹ️ ESM tokens: `[CLS] seq [EOS] [PAD]…` (`esm.py:181-182 comment`). Sequences are upper-cased/stripped by `preprocess_sequence` (`esm.py:431`).

---

## 2. Layer indexing → "layer 4" (PLAN §6.1 HF hidden-state note, D2)

✅ **Confirmed exactly as the plan's note.** InterPLM "layer L" = `hidden_states[L]` = **output of transformer block L** (1-indexed), with `hidden_states[0]` = input embeddings.
- Embedder: `outputs.hidden_states[layer]` with `output_hidden_states=True` (`esm.py:174-179`).
- Steering path independently confirms it: `orig_output.hidden_states[hidden_layer_idx]` (`intervention.py:52-53`), and the NNsight patch note "layers are indexed 0-5 in the list and we refer to them 1-6" patches the **input of `encoder.layer[hidden_layer_idx]`** (= output of block `hidden_layer_idx`) (`intervention.py:30-35`).

So **layer 4 = `hidden_states[4]` = residual stream after block 4**, byte-identical to the SAE's input. No off-by-one. ℹ️ Script *defaults* differ cosmetically (`extract_embeddings.py:20` default `[6]`; `embed_annotations.py:21` default `3`) — irrelevant, we pass `layer=4`.

---

## 3. SAE loader + dictionary (PLAN §2.2, D6)

✅ `interplm.sae.inference.load_sae_from_hf(plm_model, plm_layer, unnormalized=False)` exists (`inference.py:66`); downloads `Elana/InterPLM-{plm_model}` file `layer_{L}/ae_normalized.pt` (`inference.py:91-94`).
✅ HF layer availability hardcoded: `esm2-8m → [1,2,3,4,5,6]`, `esm2-650m → [1,9,18,24,30,33]` (`inference.py:24-27`). Matches plan.
✅ `dict_size`/`activation_dim` are read from the weight shapes on load (`dictionary.py:249`, `:515`, `:780`), so the 10,240 (=320×32) dict size is materialized from the file, not hardcoded — consistent with the plan's "don't hardcode."
ℹ️ **Parity subtlety for GLP:** the released SAE applies its **own** input normalisation internally (`normalize_to_sqrt_d` √d scaling, `dictionary.py:46-62`) at `encode` time. Its stored *input* is the **raw** `hidden_states[4]` (CLS/EOS removed). Therefore GLP must train on that **same raw activation** and do its **own** standardisation (mean/var) separately — do not feed GLP the SAE's √d-normalised tensor. Hook parity = raw residual stream, confirmed.
ℹ️ Encoding a feature subset for probing/F1: `sae.encode_feat_subset(x, feat_list, normalize_features=True)` (`dictionary.py:233` etc.); released weights are ReLU SAEs by default (`inference.py:16` imports `ReLUSAE`; class chosen from the bundled dummy config).

---

## 4. Meta-neuron tensor + timestep conditioning + block norm (PLAN §2.1, §6.2, §6.5, H2, FiLM flag)

This is the highest-value part of the recon; three plan items are refined by the actual code (`generative_latent_prior/glp/denoiser.py`, `glp/script_probe.py`).

The real block `TransformerMLPBlock.forward` (`denoiser.py:113-127`):
```python
resid_x = x
post_ln_x = self.ln(x)                 # self.ln = nn.LayerNorm(d_model)   (denoiser.py:111)
interm_x  = self.up_proj(post_ln_x)    # "up"      d_model -> d_mlp
g         = self.gate_proj(post_ln_x)  # gate pre-activation
t_emb     = self.time_proj(t_emb)      # time_proj: Linear(d_model -> d_mlp)
merged    = g * t_emb                  # multiplicative timestep modulation of the GATE
x         = self.act(merged) * interm_x  # SiLU(gate * time_proj(t)) * up   <-- meta-neuron
x         = self.down_proj(x)
return x + resid_x
```

⚠️ **DELTA D3 — LayerNorm, not RMSNorm.** Plan §6.2 says "x → RMSNorm → SwiGLU". The released block uses `nn.LayerNorm(d_model)` (`denoiser.py:111`). **Fix:** implement LayerNorm to match; RMSNorm would be an unflagged deviation from the released model.

⚠️ **DELTA D4 — timestep conditioning is plain multiplicative on the gate pre-activation, not standard FiLM.** Plan §6.2 defaults to `gate ← gate(x)·(1+γ(t)) + β(t)` and flags the exact scheme as "an extrapolation." The released scheme is now known and is simpler: a **`time_proj = Linear(d_model→d_mlp)`** applied to the sinusoidal+MLP time embedding, multiplied element-wise into the **gate pre-activation** (`denoiser.py:109,119-124`). **No `(1+γ)` scale, no `β` bias, and the `up` path is not modulated.** **Fix:** implement exactly `SiLU(gate_proj(x) * time_proj(t_emb)) * up_proj(x)`; remove the FiLM-extrapolation caveat and replace §2.1/§6.2 with the real formula.

⚠️ **DELTA D5 — meta-neuron formula must include the time factor.** Plan §2.1/§6.5 state the meta-neuron is `SiLU(gate(x))·up(x)`. The probe actually hooks the **input to `down_proj`** at each block (`script_probe.py:128` `layer_prefix="denoiser.model.layers.{i}.down_proj", location="input"`; read at `script_probe.py:104` via `TraceDict(..., retain_input=True)`). That tensor is `SiLU(gate_proj(x) · time_proj(t_emb)) · up_proj(x)` — i.e. **the time modulation is inside the SiLU.** The plan's **direction is right** (H2: post-gating gated-MLP neuron, *not* the bare gate pre-activation ✓), but the **written formula drops the `time_proj(t_emb)` factor.** **Fix:** update §2.1/§6.2/§6.5 to the down_proj-input tensor; the meta-neuron is a function of the probing time `u`. Scaffold exposes exactly this tensor from `SwiGLUBlock` (`return_meta_neuron=True`).

ℹ️ Meta-neuron **count** = `n_layers × d_mlp` (one value per d_mlp unit per block) ✅ (matches Table 12; e.g. 6×16384=98,304 for Llama8B).

---

## 5. Width rule + hyperparameters (PLAN §6.2, §2.1)

✅ Confirmed from `configs/train_llama8b_static.yaml:22-24`: `d_input=4096`, `d_model=8192 (=2×d_input)`, `d_mlp=16384 (=4×d_input)`, `n_layers=6`. So for ESM-2-8M `d_input=320 → d_model=640, d_mlp=1280`; 6 blocks → 7,680 meta-neurons. ✅
✅ Training hparams (`train_llama8b_static.yaml:34-45`): bf16, lr 5e-5, batch 4096, grad-clip 1, cosine w/ warmup_ratio 0.01, initial_factor 0.01, final_factor 0.1, 1 epoch. These are d=4096 priors — re-tune at d=320 as PLAN §6.2 already says. ✅
ℹ️ In/out projections: `in_proj: d_input→d_model`, `out_proj: d_model→d_input` (`denoiser.py:152-153`). Time embed = sinusoidal → 2-layer MLP with SiLU (`denoiser.py:76-92,155-159`).

---

## 6. Flow matching + sampler + normalizer (PLAN §2.1, §6.3)

✅ **Objective matches** (`flow_matching.py:32-35`): `noisy = (1−σ)·z0 + σ·noise`; `target = noise − z0`; MSE (`denoiser.py:285`). Here **σ (sigma) plays the role of the plan's `t`** (σ=1 pure noise, σ=0 clean).

⚠️ **DELTA D6 — sampler uses diffusers `FlowMatchEulerDiscreteScheduler.step`, not a hand-rolled `linspace(t_start,0)` Euler.** (`flow_matching.py:9,49-73`; `sample_on_manifold` `:75-104`). For a **constant** velocity field `v=ε−z0` this is mathematically exact regardless of step count (since `z(σ)=z0+σ·v` ⇒ `dz/dσ=v`), so our own linspace-Euler sampler reconstructs `z0` from `z1` exactly — the round-trip test holds either way. **Fix/decision:** implementing our own Euler in σ/t is fine and simpler for the PoC; note that to reproduce GLP's *published FD magnitudes* we'd want to match their scheduler's σ spacing (non-blocking; relevant only in W2).

⚠️ **DELTA D7 — normalizer stores `mean` & `var` (not `std`).** `normalize = (x−mean)/√var`, `denormalize = x·√var + mean` (`denoiser.py:38-46`), persisted as `rep_statistics.pt` with keys `{mean, var}` (`denoiser.py:69-71`). Plan §6.2 "zero-mean/unit-var" ✅ conceptually; **fix:** our `scaler.py` should either store `(mean, var)` in the same format (so GLP-released stats could be loaded) or clearly document `std=√var`. Scaffold stores both and matches this key layout.

---

## 7. Probing protocol + probing time (PLAN §6.5, D9)

✅ Protocol matches (`script_probe.py`): prefilter **top-k=512** candidates by `|mean(X|y=1) − mean(X|y=0)|` (`:65-80`); per-candidate **1-D LogisticRegressionCV** with `Cs=[1e-5…1e0]`, `cv=5`, `scoring='roc_auc'`, `max_iter=1000`, `penalty='l2'`, wrapped in `StandardScaler` (`:28-53`); **select by val AUC, report test AUC** (`:51-52, :145-147`); 95% bootstrap CIs over tasks (`:152-167`). ✅ Matches PLAN §6.5 headline protocol and D9.

⚠️ **DELTA D8 — probing time is `u`, released default `0.9`, in schedule-fraction space (not the plan's `t=0.1/0.5`).** `ScalarProbingConfig.u = 0.9` (`script_probe.py:211`); `u` indexes the scheduler (`fm_prepare` `:22-26`: `indices=(u·len(timesteps))`), so **larger `u` ⇒ later in the schedule ⇒ smaller σ ⇒ lightly-noised / nearer-clean** input. The plan's "t=0.5 cleanest" intuition is right but the **variable and grid must be pinned in `u`/σ space**, and the plan's numeric `t` values are not the code's `u` values. **Fix:** D9's "select on val, report on test" stays; run the selection over a `u` grid (e.g. {0.1,0.5,0.9}) and record the corresponding σ. Document the u↔σ mapping in the paper.

---

## 8. Steering / SDEdit (PLAN §2.1, §6.6)

✅ **GLP SDEdit cleanup** (`script_steer.py:11-45`): normalize → add noise at **`u=0.5`** → `sample_on_manifold(..., num_timesteps=20, start_timestep=…)` → denormalize; operates on **single-token** vectors (`(b s) 1 d`), each position cleaned independently (`:19-21`). Matches PLAN §6.6 (`t_start=0.5`, 20 steps, per-position). ✅
✅ **InterPLM steering** = NNsight decompose→clamp→recombine→continue on `EsmForMaskedLM` (`intervention.py`). The GLP add-on is `act[:, -1] = postprocess(act[:, -1] + α·w)` (`script_steer.py:69`) — additive edit then SDEdit clean, exactly the plan's recipe. ✅
ℹ️ GLP's own steering harness (`script_steer.py`) drives an **autoregressive `hf_model.generate`** loop (LLM-specific). For ESM we keep InterPLM's NNsight recon+error path and only borrow `postprocess_on_manifold` — consistent with PLAN §2.3/§6.6.

---

## 9. Swiss-Prot annotation / eval-set pipeline (PLAN §2.2, §6.1, §7)

Pipeline (README §3, verified in code): `extract_annotations` → `embed_annotations` → SAE normalize → `prepare_eval_set` → F1.

⚠️ **DELTA D9 — concept frequency-filter thresholds differ from the plan.** `make_eval_subset` defaults are `min_aa_per_concept=1500`, `min_domains_per_concept=25`, applied with strict `>` (`prepare_eval_set.py:21-22, 89-94`). PLAN §2.2/§7 say ">**10** domains OR >1,500 AAs." The `10` in README Option A is `--min_required_instances 10`, a **different** parameter — the min count for a *subcategory* to become a column at all (`extract_annotations.py:255-256,166-167`, default 100). **Fix:** PLAN §7 step 1 should read ">**25** domains OR >1,500 AAs" (or set our own explicit thresholds and log them); do not conflate with `min_required_instances`.

⚠️ **DELTA D10 — concept column names are UniProt display names + subcategory expansions, not `ft_*` codes.** The `ft_act_site`, `ft_binding`, `ft_disulfid`, `ft_helix`… in the plan are UniProt **REST download field IDs** (README:163). After `extract_annotations`, the materialized concept columns (read from `uniprotkb_aa_concepts_columns.txt`, `extract_annotations.py:239-242`) are human-readable and per-subcategory, e.g. `Active site_<note>`, `Binding site_<ligand>`, `Helix`, `Beta strand`, `Turn`, `Coiled coil`, `Disulfide bond`, `Glycosylation_<note>`, `Modified residue_<note>`, `Domain_<note>`, with `_binary` and ` [FT]` stripped (`extract_annotations.py:228-229`; column sets in `concept_constants.py:1-31`). **Fix:** `concept_selection.py` must read the materialized column file (never hardcode `ft_*`); PLAN §2.2/§7 should give the REST-field → column-name mapping.

⚠️ **DELTA D11 — eval proteins are pre-filtered to those WITH AlphaFoldDB structures, length ≤ 1022.** `preprocess_proteins` drops rows where `AlphaFoldDB` is null and keeps `Length ≤ 1022` (`extract_annotations.py:149-153, 256`); the README REST query requests `xref_alphafolddb` and `length:[1 TO …]` (README:163). **Consequences:** (a) PLAN §6.7's "AFDB-fetch + DSSP" candidate signal is **cheaper than billed** — *every* eval protein already has an AFDB accession, so the optional biophysical path needs no extra selection; consider promoting it from "only if slack." (b) The length cap is **1022** (CLS+EOS → 1024 tokens), not "<1024" as §2.2 says — cosmetic, but pin it.

⚠️ **DELTA D12 — no prebuilt 50k/25k/25k split, and the shipped 1k subset is NOT in this clone.** The split is defined by **shard ranges** (`prepare_eval_set.py:15-22` `valid_shard_range`/`test_shard_range`), not a fixed 25k/25k. The README "Option A" file `data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz` is **gitignored** (`.gitignore`: `*/data/*`, `*.gz`) and absent here (`git ls-files data/*` → 0; no `data/` dir). **Fix / W1 action:** for the **Mac PoC** obtain a small labeled set without the 12 GB UniRef50 by either (a) fetching the 1k subset file from the upstream InterPLM GitHub, or (b) running the README REST query for a small reviewed+AFDB Swiss-Prot slice, then `extract_annotations --n_shards N` and `prepare_eval_set` with chosen val/test shard ranges. Both are small downloads (ask-before-download still applies, but neither is "large").

ℹ️ **Label↔activation alignment** (for the eval store): `extract_annotations` explodes each protein to AA level in shard/protein order and one-hot/instance-encodes concepts into `aa_concepts.npz` (uint32, value = domain-instance index; `extract_annotations.py:223-246`); `embed_annotations` embeds the same shards' sequences in the same order with CLS/EOS removed (`embed_annotations.py`, using the same `ESM` embedder). Row *i* of the flattened activations ↔ AA *i* of the concatenated per-shard sequence ↔ row *i* of `aa_concepts.npz`. This ordering is the parity contract for probing.

---

## 10. GLP training-data source (PLAN §6.1, §13 Q3)

ℹ️ README states the paper trained SAEs on a random UniRef50 subset but **"Swiss-Prot… also works for training SAEs"** (README:69). For **GLP's unlabeled training store on the PoC**, a Swiss-Prot subset (or the InterPLM shard FASTAs) is a legitimate, small stand-in for UniRef50 — no 12 GB download needed for the Mac PoC. Cluster spine can still stream a UniRef50 subset later (PLAN §6.1). ⚠️ Any UniRef50 download is gated on human approval (per working agreement) — not done this session.

---

## 11. Could not verify offline (deferred to W1 on first model download)

ℹ️ ESM-2-8M `config.json` facts in PLAN §2.2 (`intermediate_size=1280`, GELU, rotary, `vocab_size=33`, `mask_token_id=32`, `pad_token_id=1`) were **not re-verified** — the HF model is not downloaded and this session does no downloads. `d=320` is corroborated by `ESM.MODEL_DIMS["facebook/esm2_t6_8M_UR50D"]=320` (`esm.py:25`); the "raw MLP neuron" baseline dim `d_ff=1280` is consistent with a 4× FFN at d=320. The hook-parity test (Deliverable C) is the mechanism that will confirm the arch facts the moment weights are present.

---

## 12. Summary of proposed PLAN.md edits

| ID | Section(s) | Change |
|----|-----------|--------|
| D1 | §2.2, §5, §6.1, D2 | Embedder is `interplm.embedders.esm.ESM` / `get_embedder("esm", …)`; `embed_single_sequence(sequence, layer)`; not `interplm.esm.embed`. |
| D2 | §5, §6.1 | Embedder already drops CLS/EOS; wrapper must not double-drop; hook-parity is exact. |
| D3 | §6.2 | Block uses **LayerNorm**, not RMSNorm. |
| D4 | §2.1, §6.2 | Timestep cond. = `SiLU(gate_proj(x)·time_proj(t_emb))·up_proj(x)` (plain multiplicative on gate pre-act; no (1+γ)/β). Remove FiLM-extrapolation caveat. |
| D5 | §2.1, §6.2, §6.5 | Meta-neuron = **input to `down_proj`** = the above tensor (includes time factor), read at probing `u`. |
| D6 | §6.3 | Reference sampler = diffusers `FlowMatchEulerDiscreteScheduler`; our Euler is exact for constant v; match σ-spacing only to reproduce FD magnitudes. |
| D7 | §6.2 | Normalizer persists `{mean, var}`; `std=√var`. |
| D8 | §2.1, §6.5, D9 | Probing time is `u` (default 0.9), schedule-fraction space; grid/report in `u`/σ, not `t=0.1/0.5`. |
| D9 | §2.2, §7 | Concept filter defaults = `>25 domains OR >1500 AAs`; `min_required_instances`(=10 in README) is a different knob. |
| D10 | §2.2, §7 | Concepts are UniProt display-name columns + subcategory expansions; read `uniprotkb_aa_concepts_columns.txt`; don't hardcode `ft_*`. |
| D11 | §2.2, §6.7 | Eval proteins all have AFDB structures (biophysical signal cheaper); length cap 1022. |
| D12 | §6.1, §9, §13 | No prebuilt 25k/25k (shard-range split); shipped 1k subset absent from clone → fetch it or REST a small Swiss-Prot slice for the PoC; no large download. |

**Nothing found that blocks the scaffold or the two tests.** The deltas are about names, one tensor formula (time factor), one norm choice, two threshold/naming corrections, and data provisioning for the PoC — all handled in the scaffold below.
