# `local_clean` — canonical run summary

Mid-scale **local** run (Apple M4, MPS, fp32) that produced the project's first results
for Deliverable D2. Config: `configs/local_clean.yaml`.

This supersedes an earlier run (`local_mid`) whose training set overlapped the eval set
by 98.4%; see §7. All numbers below come from the **decontaminated** run: the GLP
training slice has **zero** protein overlap with the eval set.

Artifacts: `probing_auc_104concepts.json`, `probing_f1_headline14.json`,
`fd_vs_steps.json`, `run_metrics.json` (all in this directory; `runs/` is gitignored).

---

## 1. What was run

| stage | detail |
|---|---|
| **Hook point** | ESM-2-8M residual stream after block 4 (`hidden_states[4]`), cls/eos dropped — byte-identical to what InterPLM's released SAE saw (guarded by `tests/test_hook_parity.py`) |
| **Training data** | 110,000 Swiss-Prot sequences via UniProt REST, **all 20k eval proteins excluded** → 10,000,000 activations, fp16 memmap (6.4 GB) |
| **GLP model** | 6 SwiGLU blocks, d_model 640, d_mlp 1280 → **20,927,680 params**, 7,680 meta-neurons |
| **Training** | flow matching, 1 epoch, 19,531 steps, batch 512, lr 5e-5 cosine+warmup |
| **Eval set** | 20,000 densely-annotated Swiss-Prot proteins → 2.53M labeled residues, 224 concepts, 110 passing the §7 stability filter, **104 probed** |
| **Probing** | 4 methods × 104 concepts (AUC) and × 14 frozen headline concepts (F1) |

Concept set was **frozen before probing** (PLAN §7) using only InterPLM's frequency
filter, split statistics, and the SAE's own tractability — never any GLP output.

## 2. Training

Loss **2.390 → 1.0371** (mean of last 200 steps), well below the ~2.0 trivial-predictor
floor for standardized flow-matching MSE. Still drifting down slightly at the end — not
fully converged. (The 2-block/2M PoC reached 1.236, so depth + 5× data bought ~0.20.)

## 3. Generation quality

**FD depends strongly on sampler steps** — quote it with its step count:

| sampler steps | FD | ratio to lower bound | spread match |
|---|---|---|---|
| 20 | 15.32 | 2.88× | 0.835 |
| 100 | 9.30 | 1.75× | 0.918 |
| 300 | 8.50 | 1.60× | 0.933 |
| **1000** | **8.24** | **1.55×** | **0.938** |

Lower bound FD(real, real) = 5.32. At 20 steps FD reports *sampler error*, not model
quality; it saturates by ~300. Config separates `sampler.num_steps` (20, steering/PCA)
from `sampler.fd_num_steps` (1000, FD).

**At 1000 steps:** ratio **1.55×**, better than GLP's own published Llama8B (5.93 vs
2.60 = 2.28×). PCA: generated captures 13.6% of variance in real's top-2 PCs vs real's
own 12.5%, centred to within 0.03 PC-std, 93.8% of real's spread.

## 4. Probing — AUC (104 concepts) — the headline result

*Can a single unit, alone, identify a per-residue biological property? Select on val, report on test.*

| method | mean test AUC | 95% CI |
|---|---|---|
| **GLP meta-neurons** | **0.9448** | [0.9315, 0.9572] |
| raw ESM MLP neuron | 0.9163 | [0.9011, 0.9307] |
| InterPLM SAE | 0.9000 | [0.8796, 0.9190] |
| raw layer output | 0.8612 | [0.8436, 0.8777] |

Paired bootstrap on per-concept differences (the correct test — matched concepts):

| comparison | mean Δ | 95% CI | wins | verdict |
|---|---|---|---|---|
| GLP − SAE | +0.0448 | [+0.0336, +0.0557] | 84/104 | significant |
| GLP − raw MLP | +0.0285 | [+0.0214, +0.0356] | 91/104 | significant |
| raw MLP − SAE | +0.0163 | [+0.0051, +0.0278] | 61/104 | significant |

Probing time `u=0.9` selected 93/104 — independently reproducing GLP's released default,
which we never hardcoded. Winning features are diverse (76 distinct of 104 concepts for
GLP), i.e. not one super-feature.

**Per-category (§7 level c):**

| category | n | GLP | SAE | GLP−SAE |
|---|---|---|---|---|
| binding_catalytic | 2 | 0.825 | 0.721 | **+0.104** |
| structural | 4 | 0.765 | 0.669 | **+0.096** |
| domain_motif | 86 | 0.954 | 0.908 | +0.046 |
| chemical_ptm | 9 | 0.968 | 0.954 | +0.013 |
| topology_localization | 3 | 0.936 | 0.938 | −0.002 |

Gains concentrate where the SAE is weakest. The pool is heavily skewed to
`domain_motif` (86/104), so the aggregate largely reflects that bucket — the
per-category table is the honest read.

## 5. Probing — F1 (14 frozen headline concepts)

InterPLM-native domain-adjusted F1 (precision per residue, recall per **domain**), at
natural prevalence on the full split. 9 AA-level, 5 domain-level.

| method | mean test F1 | precision | recall |
|---|---|---|---|
| InterPLM SAE | **0.2863** | **0.347** | 0.318 |
| GLP meta-neurons | 0.2638 | 0.241 | 0.329 |
| raw ESM MLP neuron | 0.1857 | 0.154 | 0.342 |
| raw layer output | 0.1342 | 0.099 | 0.301 |

| comparison | mean Δ | 95% CI | wins | verdict |
|---|---|---|---|---|
| SAE − GLP | +0.0225 | [−0.0079, +0.0602] | 8/14 | **NOT significant** |
| GLP − raw MLP | +0.0781 | [+0.0355, +0.1294] | 12/14 | significant |
| SAE − raw MLP | +0.1006 | [+0.0558, +0.1486] | 12/14 | significant |

`⚠ FAIRNESS` deviation (approved): operating points are chosen by **activation rate**
(per-feature quantiles) and are **two-sided**, replacing InterPLM's fraction-of-max
grid. That grid presupposes sparse non-negative ReLU features; on dense signed
activations it degenerates (56% of residues fire at thr=0, 0.07% at thr=0.5, nothing
between) and cannot see negatively-encoded concepts. F1 and recall-per-domain
*definitions* are unchanged. GLP is also given *less* freedom here than under AUC
(fixed u=0.9, no selection over the grid) — conservative for GLP.

## 6. Interpretation

**GLP leads decisively on ranking power (AUC); the SAE retains a precision advantage
that does not reach significance on F1 at n=14.**

- **AUC rewards ranking.** SAE features are **99.0% exactly zero** (median 101 active of
  10,240 per residue), so a 1-D probe on one is nearly a binary detector with no dynamic
  range. Dense features grade every residue and rank better.
- **Precision is where sparsity pays.** Rate-matching forces near-identical recall
  (0.30–0.34), so F1 differences are precision differences: SAE **0.347** vs GLP 0.241.
  The SAE fires rarely but correctly — which is the monosemanticity claim, and this
  result *supports* it.
- **But the F1 gap is not significant** (n=14, CI spans zero). Do **not** claim the SAE
  "wins on F1" from this evidence; the defensible statement is that the SAE is more
  precise while GLP is a better ranker, and that GLP significantly beats *both* raw
  baselines on F1 too.

Three consequences:

1. **The SAE is not the strongest AUC baseline** — ESM's own FFN neurons beat it
   (+0.016, significant but small, 61/104). "GLP beats the SAE" understates what
   matters; GLP beats *all* baselines.
2. **Dense-vs-sparse is a confound in GLP-vs-SAE**, since GLP meta-neurons are dense
   too. The clean control is **GLP vs raw MLP — both dense** — where GLP wins AUC by
   +0.029 (91/104) *and* F1 by +0.078 (12/14). That is the most informative comparison.
3. **AUC does not measure interpretability.** InterPLM's claim is monosemanticity; these
   results are consistent with it.

## 7. Train/eval contamination — found and fixed

The earlier `local_mid` run drew training and eval sets from two REST queries that both
matched `reviewed:true` Swiss-Prot with overlapping lengths: **19,688 of 20,000 eval
proteins (98.4%) were also in GLP's training set.** Not label leakage (GLP trains
unsupervised; the probe's own val/test protein split was always disjoint), but a
representation-level advantage GLP had and the SAE — trained on UniRef50 — did not.

Fixed by `fetch_uniprot_fasta_slice(..., exclude=...)` +
`extract_activations.py --exclude-from <eval tsv>`, filtering **during** REST pagination
so the full 110k sequences are still collected. Verified 0 overlap.

**Measured effect on AUC:** GLP 0.9472 → 0.9448, i.e. **−0.0024, 95% CI
[−0.0055, +0.0007] — not significant.** The GLP−SAE margin moved +0.0472 → +0.0448
(~5%). Every AUC conclusion survived. Consistent with the mechanism: one unsupervised
epoch over 10M activations learns the distribution, not individual vectors.

*(F1 moved more — GLP 0.2314 → 0.2638 — but with n=14 and a wide CI this is noise, not
evidence that decontamination helps F1. It is what tipped SAE−GLP below significance.)*

## 8. Limitations

1. **Local scale.** 10M activations, 6 blocks, 1 epoch, not fully converged. Cluster
   runs (50–300M), ESM-2-650M and ESMFold untested.
2. **One shard per split** (~630k residues each) — a RAM/disk concession. Positive
   counts are ~¼ of what §7's ≥500 filter assumed on the full test split.
3. **F1 is n=14** vs AUC's n=104, and its key comparison is not significant. Extending
   F1 to all 104 concepts (~3.5 h locally) is the obvious next step.
4. **Category imbalance**: 86/104 concepts are `domain_motif`; other buckets have 2–9.
5. **Eval set is 20k proteins**, not InterPLM's 50k, length-filtered 50–400 aa.
6. **Residual distribution confound**: GLP trains on Swiss-Prot, the SAE was trained on
   UniRef50. Proteins are now disjoint but the distributions still differ. Equalizing
   means training GLP on a UniRef50 slice (cluster-side).
7. `%Loss-Recovered` and the SAE-reconstruction FD anchor (**Gate G2**) not yet run —
   both are local-feasible.

## 9. Next steps

- **W2/G2**: `%Loss-Recovered` + SAE-reconstruction FD anchor.
- Extend F1 to all 104 concepts to settle the SAE-vs-GLP precision question.
- Re-run probing at full shard count for paper-grade positive counts.
- §6.5.1 nonlinearity falsification test: is a winning GLP meta-neuron linearly
  predictable from the SAE dictionary?
- **W4 steering** (glycine testbed is trivially local).

## 10. Reproduce

```bash
.venv/bin/python scripts/build_swissprot_eval.py --config configs/local_clean.yaml --n-prot 20000 --min-required-instances 30
.venv/bin/python scripts/extract_activations.py  --config configs/local_clean.yaml --store train --n-seqs 110000 --exclude-from runs/local_clean/swissprot/swissprot_annotated_eval.tsv
.venv/bin/python scripts/train_glp.py            --config configs/local_clean.yaml
.venv/bin/python scripts/eval_generation.py      --config configs/local_clean.yaml --fd-steps 20,100,300,1000
.venv/bin/python scripts/freeze_concepts.py      --config configs/local_clean.yaml --eval-dir <eval>/swissprot/processed
.venv/bin/python scripts/run_probing.py --config configs/local_clean.yaml --eval-dir <eval>/swissprot/processed --concepts <eval>/swissprot/processed/frozen_concepts.json --n-shards-per-split 1 [--metric f1 --max-concepts 14]
```

Training checkpoints every 2k steps and probing checkpoints every `(concept, method)` —
both resume automatically. Figures: `scripts/make_report.py`.
