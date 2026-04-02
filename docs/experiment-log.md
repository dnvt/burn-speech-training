# Experiment Log

35 experiments training SpeechAligner on SpeechOcean762, evaluated by Spearman ρ
against human pronunciation scores.

---

## Round 1: Baseline CTC GOP (1 run)

**Goal**: Test whether log-probability-based GOP scoring can rank pronunciation
quality from a CTC-trained model.

| Method | ρ | 95% CI |
|--------|---|--------|
| GOP LogProb | 0.106 | [0.084, 0.126] |
| GOP MaxLogit | 0.093 | [0.071, 0.114] |
| GOP Margin | 0.092 | [0.069, 0.113] |
| GOP Self-Aligned | 0.082 | [0.061, 0.104] |

**Result**: GOP scoring math alone cannot bridge the gap. The model needs
supervised pronunciation labels.

---

## Round 2: Hyperparameter Tuning (2 runs)

**Goal**: Find stable learning rate for dynamic batching.

| Config | LR | Loss | ρ |
|--------|-----|------|---|
| High LR | 0.001 | 463.4 (stalled ep3) | 0.046 |
| **Resumed, lower LR** | **0.0003** | **113.9** | **0.106** |

**Result**: LR must scale down with batch size. Dynamic batching changes
effective batch size per step — LR 0.001 diverges, 0.0003 converges.

---

## Round 3: Scoring Head Fine-Tuning (1 run, 20 epochs)

**Goal**: Add a pronunciation scoring MLP (512→256→1) trained on 25,477
human-labeled words from SpeechOcean762.

| Epoch | Total Loss | CTC | MSE | Holdout ρ |
|-------|-----------|-----|-----|-----------|
| 1 | 19.31 | 38.60 | 0.024 | — |
| 4 | 15.19 | 30.36 | 0.023 | 0.221 |
| 10 | 10.17 | 20.32 | 0.022 | 0.220 |
| 18 | 5.29 | 10.56 | 0.021 | — |

**Result**: ρ plateaus at ~0.22 from epoch 4. CTC loss keeps decreasing (the
model improves at alignment) but MSE barely moves (scoring doesn't improve).
Root cause: 90.8% of SpeechOcean762 samples score 10/10 — the model predicts ~1.0 for
everything.

---

## Round 4: Loss Ablation (13 runs)

**Goal**: Break through the 0.22 plateau by changing the loss function.

| Run | Config | ρ |
|-----|--------|---|
| 4a | CTC=0.5, MSE=0.5 (baseline) | 0.221 |
| 4b | CTC=0.3, MSE=0.7 | 0.237 |
| 4c | CTC=0.1, MSE=0.9 | 0.261 |
| **4d** | **CTC=0.0, MSE=1.0** | **0.287** |
| **4e** | **CTC=0.0, MSE + warmup** | **0.292** |
| 4f | CTC=0.0, focal loss | 0.264 |
| 4g | CTC=0.0, weighted MSE | 0.271 |
| 4h | CTC=0.0, larger head (512) | 0.285 |
| 4i-4m | Various focal/weighting combos | 0.258-0.279 |

**Key finding**: CTC weight = 0 is the dominant improvement (+0.07 ρ over
baseline). CTC gradient actively interferes with the scoring head. Warmup adds
a small but consistent +0.005 ρ.

**Ruled out**: focal loss (hurts), weighted loss with CTC=0 (hurts), larger
head (no effect).

---

## Round 5: Schedule Search (5 runs)

**Goal**: Optimize training schedule around the CTC=0 recipe.

| Run | Config | ρ |
|-----|--------|---|
| 5a | CTC=0, warmup=3, cosine decay | 0.292 |
| 5b | CTC=0, frozen backbone (first 5 epochs) | 0.258 |
| 5c | CTC=0, warmup=5 | 0.291 |
| **5d** | **CTC=0, warmup=3, cosine, lower LR** | **0.292** |
| 5e | CTC=0, frozen backbone entire training | 0.241 |

**Result**: The 0.292 ceiling is reproducible across schedule variants. Freezing
the backbone hurts significantly. Warmup + cosine is the stable recipe.

---

## Round 6: Architecture Search (13 runs)

**Goal**: Break through 0.29 with structural changes to the model or objective.

| Run | Approach | ρ |
|-----|----------|---|
| 6a-6b | Knowledge distillation | 0.292 |
| 6c | Distillation + rank regularization | 0.288 |
| 6d | Pooled-feature rank regularization | 0.288 |
| 6e | True ordinal softmax CE | 0.283 |
| 6f | Weighted ordinal softmax | 0.279 |
| 6g-6h | Attention pooling (2 variants) | 0.283, 0.282 |
| 6i-6m | Combined approaches | 0.275-0.288 |

**Result**: None exceeded the 0.292 baseline. The ≈0.29 ceiling is a
representation quality bottleneck, not a loss or architecture problem. Richer
input features are the path forward.

---

## Summary

| Round | Best ρ | Experiments | Insight |
|-------|--------|-------------|---------|
| 1. Baseline GOP | 0.106 | 1 | Log-prob scoring alone can't rank pronunciation |
| 2. Hyperparameters | 0.106 | 2 | LR must scale with batch size |
| 3. Scoring head | 0.221 | 1 | Supervised scoring reaches 0.22, then plateaus |
| 4. Loss ablation | 0.292 | 13 | CTC=0 is the single biggest gain |
| 5. Schedule search | 0.292 | 5 | Ceiling is reproducible across schedules |
| 6. Architecture | 0.288 | 13 | Structural changes don't break through |
| **Total** | **0.292** | **35** | **Bottleneck is feature representation** |
