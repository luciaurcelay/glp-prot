# `local_mid` — run summary

Mid-scale **local** run (Apple M4, MPS, fp32) that produced the project's first real
results for Deliverable D2. Config: `configs/local_mid.yaml`.

> `runs/` is gitignored, so this file is not in version control. The metric artifacts
> behind every number here are copied to `docs/results/` (`probing_104concepts.json`,
> `probing_f1_headline14.json`, `fd_vs_steps.json`, `local_mid_summary.json`).

---

## 1. What was run

| stage | detail |
|---|---|
| **Hook point** | ESM-2-8M residual stream after block 4 (`hidden_states[4]`), cls/eos dropped — byte-identical to what InterPLM's released SAE saw (guarded by `tests/test_hook_parity.py`) |
| **Training data** | 110k Swiss-Prot sequences via UniProt REST → **10,000,000 activations**, fp16 memmap (6.4 GB). No UniRef50. |
| **GLP model** | 6 SwiGLU blocks, d_model 640, d_mlp 1280 → **20,927,680 params**, 7,680 meta-neurons |
| **Training** | flow matching, 1 epoch, 19,531 steps, batch 512, lr 5e-5 cosine+warmup |
| **Eval set** | 20k densely-annotated Swiss-Prot proteins → 2.53M labeled residues, **224 concepts**, 110 passing the §7 stability filter |
| **Probing** | 4 methods × 104 concepts (AUC) and × 14 frozen headline concepts (F1) |

Concept set was **frozen before probing** (`frozen_concepts.json`, PLAN §7) using only
InterPLM's frequency filter, split statistics, and the SAE's own tractability — never
any GLP output.

## 2. Training

Loss **2.349 → 1.034** (mean of last 200 steps; the logged `final_loss` 1.001 is a
single noisy step). Well below the ~2.0 trivial-predictor floor for standardized
flow-matching MSE. Curve still drifting down slightly at the end — not fully converged.

For reference, the earlier 2-block/2M PoC reached 1.236, so depth + 5× data bought ~0.20.

## 3. Generation quality

**FD depends strongly on sampler steps** — a finding that changed how we report it:

| sampler steps | FD | ratio to lower bound | spread match |
|---|---|---|---|
| 20 | 14.77 | 2.81× | 0.849 |
| 50 | 10.38 | 1.98× | 0.907 |
| 100 | 9.13 | 1.74× | 0.928 |
| 300 | 8.37 | 1.59× | 0.943 |
| **1000** | **8.12** | **1.55×** | **0.948** |

Lower bound FD(real, real) = 5.25. At 20 steps FD reports *sampler error*, not model
quality; it saturates by ~300 steps. The config now separates `num_steps` (20, for
steering/PCA) from `fd_num_steps` (1000, for FD).

**At 1000 steps:** FD ratio **1.55×**, better than GLP's own published Llama8B (5.93 vs
2.60 = 2.28×). PCA: generated captures 13.6% of variance in real's top-2 PCs vs real's
own 12.2%, centered to within 0.03 PC-std, 94.8% of real's spread.

Scale comparison at matched 1000-step sampling: PoC (2-blk/2M) FD ratio 2.09× →
mid (6-blk/10M) **1.55×**. Loss and FD improve together.

## 4. Probing — AUC (104 concepts)

*Can a single unit, alone, identify a per-residue biological property? Select on val, report on test.*

| method | mean test AUC | 95% CI |
|---|---|---|
| **GLP meta-neurons** | **0.9472** | [0.9341, 0.9594] |
| raw ESM MLP neuron | 0.9163 | [0.9011, 0.9307] |
| InterPLM SAE | 0.9000 | [0.8796, 0.9190] |
| raw layer output | 0.8612 | [0.8436, 0.8777] |

Paired bootstrap on per-concept differences (the correct test — matched concepts):

| comparison | mean Δ | 95% CI | wins |
|---|---|---|---|
| GLP − SAE | +0.0472 | [+0.0370, +0.0572] | 86/104 |
| GLP − raw MLP | +0.0309 | [+0.0239, +0.0380] | 92/104 |
| GLP − raw layer | +0.0860 | [+0.0741, +0.0979] | 101/104 |
| raw MLP − SAE | +0.0163 | [+0.0051, +0.0278] | 61/104 |

All significant. Probing time `u=0.9` selected 88/104 — independently reproducing GLP's
released default, which we never hardcoded.

**Per-category (§7 level c):**

| category | n | GLP | SAE | raw MLP | raw layer | GLP−SAE |
|---|---|---|---|---|---|---|
| binding_catalytic | 2 | 0.825 | 0.721 | 0.794 | 0.797 | **+0.104** |
| structural | 4 | 0.770 | 0.669 | 0.703 | 0.661 | **+0.101** |
| domain_motif | 86 | 0.956 | 0.908 | 0.928 | 0.867 | +0.048 |
| chemical_ptm | 9 | 0.966 | 0.954 | 0.943 | 0.901 | +0.012 |
| topology_localization | 3 | 0.946 | 0.938 | 0.875 | 0.879 | +0.008 |

Gains concentrate where the SAE is weakest. The pool is heavily skewed to
`domain_motif` (86/104), so the aggregate largely reflects that bucket.

## 5. Probing — F1 (14 frozen headline concepts)

InterPLM-native domain-adjusted F1 (precision per residue, recall per **domain**),
evaluated at natural prevalence on the full split. 9 AA-level, 5 domain-level concepts.

| method | mean test F1 | precision | recall |
|---|---|---|---|
| **InterPLM SAE** | **0.2863** | **0.347** | 0.318 |
| GLP meta-neurons | 0.2314 | 0.195 | 0.312 |
| raw ESM MLP neuron | 0.1857 | 0.154 | 0.342 |
| raw layer output | 0.1342 | 0.099 | 0.301 |

Paired: SAE − GLP **+0.055** [+0.004, +0.111], SAE wins 9/14, GLP 4/14, 1 tie —
significant. SAE − raw MLP +0.101 [+0.056, +0.149], 12/14. GLP − raw MLP +0.046
[−0.006, +0.110] — **not** significant.

`⚠ FAIRNESS` deviation (approved): operating points are chosen by **activation rate**
(per-feature quantiles) and are **two-sided**, replacing InterPLM's fraction-of-max
grid. That grid presupposes sparse non-negative ReLU features; on dense signed
activations it degenerates (56% firing at thr=0, 0.07% at thr=0.5, nothing between)
and cannot see negatively-encoded concepts. The F1 and recall-per-domain *definitions*
are unchanged. **The conclusion is identical under both schemes**, which is what makes
it trustworthy. GLP is also given *less* freedom here than under AUC (fixed u=0.9, no
selection over the grid) — conservative for GLP.

## 6. Interpretation

**The two metrics disagree, and that disagreement is the finding.**

- **AUC rewards ranking.** SAE features are **99.0% exactly zero** (median 101 active of
  10,240 per residue), so a 1-D probe on one is nearly a binary detector with no
  dynamic range. Dense features grade every residue and therefore rank better.
- **F1 rewards precision at an operating point.** Rate-matching forces near-identical
  recall (0.30–0.34 across all methods), so the entire F1 gap is precision: at matched
  firing rate SAE features are **~1.8× more precise** than GLP meta-neurons.

These are two views of the same property. Sparse features fire rarely but correctly;
dense features rank everything well but must threshold an always-on signal.

Three consequences:

1. **The SAE is not the strongest AUC baseline** — ESM's own FFN neurons beat it
   (+0.016, significant but small, and only 61/104 concepts). So "GLP beats the SAE"
   overstates; GLP beats *all* baselines, and the informative comparison is against
   raw MLP.
2. **Dense-vs-sparse is a confound in the GLP-vs-SAE comparison**, because GLP
   meta-neurons are dense too. The clean control is **GLP vs raw MLP — both dense** —
   where GLP wins AUC by +0.031 on 92/104 and ties on F1.
3. **AUC does not measure interpretability.** InterPLM's claim is monosemanticity, and
   the F1 result *supports* it rather than undermining it.

## 7. Conclusions

- The full local pipeline works end-to-end and **Gate G1 is passed** with margin.
- **GLP meta-neurons isolate concepts more sharply than every baseline under 1-D AUC**
  (0.947 vs 0.916/0.900/0.861, all paired-significant, n=104) — Deliverable D2's core
  claim has support at local scale.
- **The SAE remains more precise** at a matched operating point. The honest framing is
  PLAN §1's *"complementary, not a competitor that must win"* — now evidence-backed
  rather than a hedge.
- GLP as a generative model is healthy: FD 1.55× the lower bound at 1000 steps, better
  than GLP's own published Llama8B ratio.

## 8. Limitations

1. **Local scale.** 10M activations, 6 blocks, 1 epoch, not fully converged. Cluster
   runs (50–300M) are untested.
2. **One shard per split** (~630k residues each) — a RAM/disk concession. Positive
   counts are ~¼ of what §7's ≥500 filter assumed on the full test split.
3. **F1 is n=14**, AUC is n=104. The F1 conclusion is weaker.
4. **Category imbalance**: 86/104 concepts are `domain_motif`; other buckets have 2–9.
5. **Eval set is 20k proteins**, not InterPLM's 50k, and length-filtered to 50–400 aa.
6. `%Loss-Recovered` and the SAE-reconstruction FD anchor (Gate G2) are **not yet run**.

## 9. Next steps

- **W2/G2**: `%Loss-Recovered` + SAE-reconstruction FD anchor (both local-feasible).
- Re-run probing at full shard count for paper-grade positive counts.
- Extend F1 to all 104 concepts (~3.5 h locally).
- §6.5.1 nonlinearity falsification test: is a winning GLP meta-neuron linearly
  predictable from the SAE dictionary? Directly tests whether GLP encodes structure
  outside the SAE's span.
- **W4 steering** (glycine testbed is trivially local).

## 10. Reproduce

```bash
.venv/bin/python scripts/extract_activations.py  --config configs/local_mid.yaml --store train --n-seqs 110000
.venv/bin/python scripts/train_glp.py            --config configs/local_mid.yaml
.venv/bin/python scripts/eval_generation.py      --config configs/local_mid.yaml --fd-steps 20,50,100,300,1000
.venv/bin/python scripts/build_swissprot_eval.py --config configs/local_mid.yaml --n-prot 20000 --min-required-instances 30
.venv/bin/python scripts/freeze_concepts.py      --config configs/local_mid.yaml --eval-dir runs/local_mid/swissprot/processed
.venv/bin/python scripts/run_probing.py --config configs/local_mid.yaml \
    --eval-dir runs/local_mid/swissprot/processed \
    --concepts runs/local_mid/swissprot/processed/frozen_concepts.json \
    --n-shards-per-split 1 [--metric f1 --max-concepts 14]
```

Probing is resumable — each `(concept, method)` checkpoints to
`metrics/probing_partial[_f1].jsonl`. Figures: `scripts/make_report.py`.
