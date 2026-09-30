# Depth Anything 3 Compression — Results

**DA3 (DINOv2 + DualDPT) · KITTI odometry · Geometry-aware knowledge distillation + P3B block pruning + TensorRT INT8 on Jetson AGX Orin**

*By [Tommy Chen (陳龍懷)](https://linkedin.com/in/tommychen19920812)*

Compressing a multi-view metric-depth model instead of a classifier, and taking it to a device.
DA3 predicts dense depth, ray maps and point maps from a 5-frame window, so every compression
decision has to be judged on geometry, not on Top-1. Every number below comes from a full-split
evaluation run, a CUDA FLOP count, or an engine timed on an idle card — never from a training-time
proxy — and differences carry paired bootstrap confidence intervals.

**Confidence intervals were not enough, and a good part of this project is about finding that
out.** Bootstrap intervals cover sampling; they say nothing about re-training. Measuring the
seed-to-seed spread directly, late, put it at 0.0054 abs_rel — larger than several differences this
document had already reported as real. Two of those claims are withdrawn below rather than
quietly dropped, and the same pattern recurred in deployment, where a misaligned width made a
whole latency ranking come out backwards. Both are marked where they occur.

---

## Status

| Stage | State |
|---|---|
| Reproduce DA3 fine-tuning on KITTI | **Done** — one recipe, used unchanged by every row |
| DA3-BASE teacher, DA3-Small student, 4 distillation variants | **Done** — §2, §3 |
| Compute-parity analysis and DualDPT head pruning | **Done** — §4 |
| P3B pruning of DA3-BASE to DA3-Small's compute | **Done on compute, lost on accuracy** — §5 |
| Closing the accuracy gap | **7 changes measured — 5 helped, 2 did not.** Gap to DA3-Small cut from 0.034 to 0.010, not closed — §5 |
| TensorRT INT8, five routes, and Jetson AGX Orin deployment | **Done** — §6. Best depth 0.1043 on Orin, *below* the PyTorch model it was quantized from |
| Run-to-run variance measured, and earlier conclusions withdrawn | **Done** — 0.0054 abs_rel between seeds, which retired two claims this document used to make

The headline result is negative: at matched compute, a pruned DA3-BASE does not reach a
DA3-Small that carries full DINOv2 pretraining. What follows is the evidence that this is a
property of the comparison rather than of the implementation — including two defects found in my
own port and one in my own baseline, which had to be fixed before the claim was worth making.

The deployment half is not negative. The pruned model reaches **22.6–25.1 ms on a Jetson AGX Orin**,
and the best INT8 route costs **+0.0003 abs_rel** against the model it was quantized from — on
Orin, nothing measurable at all.

---

## TL;DR

| | |
|---|---|
| **Compute parity reached by pruning** | 729.1 → **203.1–209.8 GFLOPs** against DA3-Small's 205.6; 101.97 M → **25.6–31.9 M** params. The best configuration is under DA3-Small on both axes |
| **Free compute found by profiling** | **31.5 GFLOPs and 1.16 M params** removed bit-exactly — the DualDPT head computes three auxiliary pyramid branches whose output is never read. DA3-Small still pays 7.9 GFLOPs for them |
| **Port defect found and fixed** | P3B's block importance had collapsed to `0.693 / param_count`, silencing the method's central signal. Restoring the official max-rescaling recovered a **4-orders-of-magnitude** importance spread |
| **Baseline defect found and fixed** | A backbone learning-rate multiplier of ×0.1, inherited from an earlier recipe, was throttling **every row**. Releasing it improved DA3-Small by **0.0062 abs_rel — 6× the total effect of all four distillation variants** |
| **Distillation is worth more after pruning than before it** | On DA3-Small, KD buys **−0.0011** abs_rel, inside that setting's ±0.0015 floor. On the pruned model it buys **−0.0045**, at least 3× the corresponding floor — pruning destroys information a teacher can restore, while a pretrained release has little left to recover. The direction is consistent; the exact ratio is not claimed, because −0.0045 sits inside the 0.0054 seed spread of pruned runs (§3) |
| **Best pruned result** | **0.1045** abs_rel at **203.1 GFLOPs / 25.63 M** — under DA3-Small on both axes — against DA3-Small's **0.0944**. Gap **+0.0101**, CI [+0.0086, +0.0116] — still a real loss, from 0.1291 at the start |
| **Latency won at the pruning stage, not the quantization stage** | P3B baked MLP widths that were not multiples of 64 (1429, 1229 …), so TensorRT sent `fc1` down a generic GEMM. Aligning them moved the FP16 engine **2.761 → 2.180 ms, −21%**, by switching 7 GEMMs to `ampere_h16816gemm_256x128`. Three builds per side, non-overlapping |
| **A ranking I had to reverse** | That same misalignment had made explicit INT8 look faster than FP16 — it penalised only the FP16 paths. Corrected, **every explicit INT8 route is slower than plain FP16 on a 4090**, and the fastest engine is TensorRT's own implicit INT8 at 1.939 ms |
| **Deployment target reached** | Same ONNX rebuilt on **Jetson AGX Orin 64 GB** (TRT 10.3): 22.60–25.13 ms across seven engines, depth agreeing with the 4090 to **0.0006**. Best on Orin is a head-only Hessian-weighted route at **0.1043 / 24.59 ms** |
| **What the gap is not** | Not the architecture P3B finds. The two models have *identical* skeletons; forcing width into the layer BPI rated lowest (block 11: 3/307 → 5/1229) moved depth by −0.0010, CI [−0.0021, +0.0001]. What moved the pruned model was training — **the fine-tune learning rate alone was worth 0.0105** |
| **Measurement discipline is most of the work** | Bootstrap CIs do not cover re-training, so the seed spread was measured directly: **0.0054 abs_rel**, because the seed changes P3B's allocation itself. That figure **retired two conclusions this document previously stated as real** (§3, §5). Separately, roughly **half of TensorRT 10.10's builds of this graph are numerically broken** — so every engine is validated on val before it counts, and latency is best-of-N on an idle card |

---

## Pipeline

```text
                DA3-BASE  (DINOv2 ViT-B + DualDPT)  101.97 M / 729.1 GFLOPs
                                  │
        ┌─────────────────────────┴─────────────────────────┐
   Route A: distil into the release        Route B: prune the big model
        │                                                   │
   DA3-BASE teacher                         P3B block pruning (ICCVW 2025)
   0.0862 abs_rel                           + grouped Fisher head pruning
        │                                   + dead-branch removal (bit-exact)
        │  geometry-aware KD                + residual-stream pruning (this work)
        │  depth L1 + log L1                              │
        │  ray cosine + origin L1                         │  then the same KD
        ▼  feature cosine                                 ▼
   DA3-Small student                        pruned DA3-BASE
   25.93 M / 205.6 GFLOPs                   25.63 M / 203.1 GFLOPs
   0.0944 abs_rel                           0.1045 abs_rel
                                                    │
                                                    │  §6: TensorRT INT8, 5 routes
                                                    ▼
                                            Jetson AGX Orin 64 GB
                                            22.60 – 25.13 ms
                                            0.1043 – 0.1075 abs_rel
```

Both routes end at the same compute; §5 is about why they do not end at the same accuracy.
§6 takes the pruned model the rest of the way to the device.

---

## Experimental Setup

| Item | Value |
|---|---|
| Dataset | **KITTI odometry, standard split** — train 00, 01, 02, 04, 06, 07, 08 / val 05 / test 09, 10. Sequence 03's raw drive is not publicly released |
| Supervision | Raw LiDAR depth — **~4.4% of pixels** carry a value |
| Sample | 5-frame window; 168 × 518 training, 8,309 windows. Inference 5 views @ 364 × 1218, `process_res` 518 |
| Models | DA3-BASE: ViT-B (12 blocks, 12 heads × 64, width 768) + DualDPT. DA3-Small: ViT-S (width 384, 6 heads, MLP 1536) + DualDPT |
| Recipe (every row) | Whole backbone unfrozen, 25 epochs, lr 5e-5 cosine, AdamW, bf16, effective batch 32. Backbone lr multiplier ×1.0 — see §2 |
| Checkpoint selection | Best **val abs_rel**. Pose AUC selects epochs with worse depth; the loss is unusable (its confidence term is negative and rising) |
| Evaluation | Depth: non-overlapping windows, test 695 / val 690 frames, per-frame median alignment, 0–80 m. Pose: from the ray head, not the camera decoder |
| Significance | Paired bootstrap over windows, 10,000 resamples. **All method decisions taken on val only.** Run-to-run noise measured directly rather than assumed — see the row below |
| Run-to-run variance | **0.0054 abs_rel** between two pruned runs differing only in `--seed`, and 0.0135 on pose AUC@3°; four same-family runs span 0.0060. The seed changes **P3B's allocation itself** (MLP total width 16320 against 16384, one block 7 heads against 8), so this is an order of magnitude above the ±0.0015 measured where the architecture is fixed. **Bootstrap intervals cover window sampling, not training variance** — two separately-trained pruned runs closer than ≈0.005 are not distinguishable, however tight their CI. Quantization routes share one set of weights and are unaffected |
| Compute accounting | CUDA `FlopCounterMode` on the inference path. A CPU count silently misses every `scaled_dot_product_attention` — exactly 42.6 GFLOPs for DA3-Small |
| Parameter accounting | backbone + head, i.e. the depth/ray path actually executed. Camera encoder/decoder excluded: training drops them and pose comes from the ray head |
| Hardware | 2 × RTX 4090 (24 GB), TensorRT 10.10. DA3-BASE fine-tuning peaks at 23.9 GB per card. Deployment target: **Jetson AGX Orin 64 GB**, JetPack 6.2.1 / TensorRT 10.3, clocks and EMC locked at MAXN |
| Latency protocol | Engines are timed **after the fact on an idle card**, never from the number `trtexec` prints during a build: the same engine measures 2.47 ms mid-build against 2.39 ms idle, and the two orderings disagree. Each route builds N engines, validates every one, and the fastest valid engine is timed — median of 3 rounds, all routes in one window |

---

## 1 · Compute Budget

| Model | backbone | head | total params | backbone | head | **total GFLOPs** |
|---|---:|---:|---:|---:|---:|---:|
| DA3-Small | 22.06 M | 3.87 M | 25.93 M | 123.3 | 82.2 | **205.6** |
| DA3-BASE | 86.58 M | 15.39 M | 101.97 M | 420.0 | 309.0 | **729.1** |

DA3-BASE is **3.5× the compute**. Note the split: the head alone is 309.0 GFLOPs — **more
than the entire DA3-Small model** — so pruning the backbone cannot reach parity on its own.
That is why §4 exists.

---

## 2 · Baselines, the Teacher, and a Throttled Learning Rate

**Test (sequences 09 + 10)**

| Model | abs_rel ↓ | rmse ↓ | d1 ↑ | AUC@3° ↑ |
|---|---:|---:|---:|---:|
| DA3-Small, pretrained | 0.1370 | 4.307 | 0.8230 | 0.161 |
| DA3-BASE, pretrained | 0.1243 | 4.069 | 0.8532 | 0.171 |
| DA3-Small, fine-tuned (backbone lr ×0.1) | 0.1014 | 3.441 | 0.8989 | 0.548 |
| **DA3-Small, fine-tuned (×1.0)** | **0.0952** | 3.333 | 0.9065 | 0.5824 |
| **DA3-BASE, fine-tuned (teacher, ×0.1)** | **0.0862** | **3.165** | **0.9210** | 0.564 |

**The baseline was throttled, and finding that mattered more than any distillation result.**
The ×0.1 backbone multiplier came from an earlier recipe and had never been validated on this
split. Releasing it to ×1.0 improved DA3-Small by **0.0062 abs_rel** — against **0.0011** for
the best of four distillation variants. Every "does KD help?" judgement made before that fix
was made against a suppressed baseline, and every earlier row had to be re-measured.

The teacher has **not** been re-run at ×1.0, so the 0.0090 teacher–student gap it now shows is
not a matched-recipe measurement and is very likely an underestimate. Stated rather than
quietly compared.

Two observations survive the correction: **fine-tuning matters more than model size** —
DA3-Small fine-tuned (0.0952) beats DA3-BASE pretrained (0.1243) at a fifth of the compute —
and **pose is not a proxy for depth**, since on val the smaller fine-tuned model has the better
AUC@3° while being clearly worse on depth. Reporting one of the two alone would mislead.

---

## 3 · Geometry-Aware Knowledge Distillation

DA3 emits dense geometry, not class probabilities, so there is no logit to soften. Every term
is a geometric residual between student and teacher on the same input window.

```
L        = L_GT + α · L_KD                         α = 1, GT weight fixed at 1
L_KD     = λ_d · L_depth + λ_r · L_ray + λ_f · L_feat
L_depth  = λ_abs · |D_S − D_T| / s  +  λ_log · |log D_S − log D_T|
L_ray    = λ_dir · (1 − cos(d_S, d_T))  +  |o_S − o_T| / s
L_feat   = 1 − cos(P(f_S), f_T)
```

| Symbol | Value | Note |
|---|---|---|
| λ_d / λ_r / λ_f, λ_abs / λ_log | 1 / 1 / 0.5, 0.5 / 0.5 | block and depth weights |
| λ_dir | 10 | ray direction — the only unit-free residual (1 − cos ∈ [0, 2]) |
| λ_pointmap | 0 | off: P = D·d + o duplicates the first two terms |
| mask | 0 < D_T ≤ 80 m | teacher-only, covering **99.95%** of pixels |
| s | mean valid GT depth of the window | so KD and GT share one unit |
| feature layer | cross-view tokens | student projected to the teacher's width by a trainable linear layer, discarded after training |

**Why a teacher-only mask**: LiDAR supervises 4.4% of pixels. Intersecting GT with the
teacher's valid region keeps **4.5%**; the teacher alone keeps **99.95%**. Both were measured
before choosing. Distillation is here to supply density, not accuracy.

**On DA3-Small, distillation does almost nothing for depth** (test, vs DA3-Small fine-tuned):

| Variant | abs_rel | AUC@3° | Δ abs_rel (CI) | Δ AUC@3° (CI) |
|---|---:|---:|---|---|
| v1 — all-L1 | 0.1005 | 0.530 | −0.0009 [−0.0013, −0.0004] | −0.018 [−0.026, −0.009] |
| v1 — convex 0.7·KD + 0.3·GT | 0.1000 | 0.529 | −0.0013 [−0.0018, −0.0008] | −0.019 [−0.029, −0.010] |
| v2 — log depth, cosine ray | 0.1006 | 0.534 | −0.0007 [−0.0012, −0.0003] | −0.014 [−0.022, −0.006] |
| **v2 + feature KD** | 0.1003 | **0.570** | −0.0010 [−0.0014, −0.0006] | **+0.022 [+0.013, +0.032]** |

- **Depth: no variant helps.** Every gain is below the ±0.0015 seed noise and recovers only
  5–9% of the teacher gap. Changing the loss form did not help either — geometry-aware v2 is
  statistically indistinguishable from plain-L1 v1.
- **Output-only distillation makes pose worse**, consistently. Adding one feature term
  reverses it to **+0.022** on test, significant at all four thresholds.
- **The diagnostic:** the feature loss fell 0.97 → 0.0137, i.e. projected student tokens
  reached **cosine 0.986** with the teacher's — and depth still did not move. The teacher's
  advantage is not a misaligned representation that distillation can transfer; it is capacity.
  **This is what motivated pruning the large model instead of training the small one.**

**On a pruned model, the same distillation is worth four times as much** (test, ×1.5 recipe):

| | abs_rel ↓ | d1 ↑ | AUC@3° ↑ |
|---|---:|---:|---:|
| DA3-BASE + P3B, plain fine-tune | 0.1097 | 0.8770 | 0.5952 |
| **DA3-BASE + P3B + KD** | **0.1052** | 0.8873 | 0.5674 |
| **KD's contribution** | **−0.0045** | +0.0103 | −0.0278 |

−0.0045 against −0.0011 on DA3-Small. The asymmetry has a clean reading: pruning removes
information the teacher still holds and can hand back, whereas a pretrained release has little
left to recover — which is what would justify distillation as a post-pruning step rather than an
alternative to it.

**How much of that survives the variance measurement, stated plainly.** −0.0045 sits just inside
the 0.0054 spread measured between pruned runs that differ only in seed (Setup), so the
*magnitude* is not established by these two runs alone; a single seed pair cannot separate it.
What does survive is the asymmetry itself: on DA3-Small, where the architecture is fixed and the
noise floor is ±0.0015, four distillation variants all landed inside that floor, whereas on the
pruned model the effect is at least three times the corresponding floor. The direction is
consistent across every pruned run measured. **The honest statement is that distillation is worth
more after pruning than before it, not that the ratio is 4×.** Pinning the ratio needs two seeds
per arm, which has not been run. (Pose differences of this size sit inside the ±0.03 bootstrap
width and are not claimed either way.)

---

## 4 · Reaching DA3-Small's Compute

P3B prunes Transformer sub-blocks; it does not cover a DPT head. Since the head is 42% of
DA3-BASE's compute, it had to be profiled and pruned too.

**Where the head's 309.1 GFLOPs are** (leaf modules, 5 views, inference size)

| Module | GFLOPs | |
|---|---:|---|
| `output_conv1_aux.3` | 96.0 | the single largest item — larger than either fusion chain |
| `refinenet*` main / aux | 56.4 / 56.4 | |
| `output_conv1_aux.0/1/2` | **31.5** | **computed and discarded** |
| `output_conv1`, `output_conv2.0` | 19.2 / 14.7 | |
| `projects*`, `resize*`, `layer*_rn` | 29.9 | |

- **31.5 GFLOPs and 1.16 M parameters removed bit-exactly.** `_fuse()` runs the auxiliary
  pre-head stack on all four pyramid levels while the forward pass reads only the last; three
  `output_conv2_aux` branches are never called at all. Replacing all six modules with
  `Identity` leaves every returned tensor unchanged — no retraining, no accuracy cost. Found
  by profiling leaf modules, not by reading the architecture diagram. **DA3-Small has the same
  dead branches and still pays 7.9 GFLOPs for them** (§5).
- The largest single cost is the *auxiliary* pre-head stack, not the fusion chain — the
  opposite of what the module names suggest.
- Head pruning needs **coupled channel groups**: DPT's fusion blocks add tensors, so the
  `features` width is one group spanning **both** the main and auxiliary chains.
- Hard constraint found by assertion failure: head widths must divide by 4, because a sincos
  positional embedding is applied to `projects.i` and `output_conv1` outputs.

**A parameter budget does not control compute.** Eq. 2 of the paper divides benefit by
parameters, and an attention sub-block is half the parameters of an MLP one while running over
`S·N` tokens in the cross-view blocks. A 29% *parameter* budget kept 45.8% of attention heads
against 20.9% of MLP neurons and landed at **212.6 GFLOPs — above DA3-Small**. Budgeting the
water-filling in FLOPs instead reaches parity while leaving Eq. 2's importance per-parameter,
exactly as the paper defines it.

---

## 5 · P3B: the Port, the Defect, and What the Pruned Model Actually Looks Like

P3B allocates each block's width from a Block Performance Indicator,
`Δψ_i = L(h_i(x_{i−1})) − L(h_i(x_i))`: how much better a lightweight probe reads the task after
the sub-block than before it. The paper's probe is a classifier; depth has no classes, so it was
replaced with a linear per-token log-depth probe trained against LiDAR pooled onto the 14 × 14
patch grid (66.2% of patches carry a return, against 4.4% of pixels).

### The defect: one missing line silences the method

| | result |
|---|---|
| `softplus(Δψ) / params` (my first port) | all 12 blocks got the **same width** — 5 of 12 heads each, MLP 581–622, a 1.07× spread |
| Official transform | importance spans **0.0000 – 7.8770** across 24 sub-blocks |
| Scale invariance check | Δψ × 0.1, × 1 and × 10 give **identical** importance |

Δψ shrinks as the probe converges — on DA3-BASE the largest sub-block benefit fell **0.51 → 0.06
over three warm-up epochs**, reproduced on every run since — because one probe trained on both
endpoints learns to read depth equally well from either. With a plain softplus every block lands
on `softplus(0) = 0.693` and the allocation degenerates into `1 / param_count`. The official
implementation rescales Δψ by its maximum before the shaping function, discarding absolute
magnitude and keeping only the pattern across blocks. The same degeneracy is visible in an
earlier MobileViTv3 port: importance there equals `0.693 / params` to three decimals.

A second correction: the soft mask belongs on the **rank** axis, not on score values. Fisher
scores are extremely long-tailed (one sub-block's max/median ratio is 2715), and on the score
axis 96% of the mask was still stuck between 0.4 and 0.6 at the sharpest setting — the sharpening
phase was doing nothing. On the rank axis the kept count is exactly `round(k·N)`.

### The result: compute parity, accuracy loss

**Test (sequences 09 + 10). Baseline is DA3-Small at the corrected ×1.0 recipe.**

| | abs_rel ↓ | d1 ↑ | AUC@3° ↑ | GFLOPs | params |
|---|---:|---:|---:|---:|---:|
| DA3-Small + KD | **0.0944** | 0.9080 | 0.5688 | 205.6 | 25.93 M |
| DA3-Small, fine-tuned | 0.0952 | 0.9065 | 0.5824 | 205.6 | 25.93 M |
| **BASE + P3B + KD, residual pruned, floor 0.4, MLP aligned to 64** (best) | **0.1045** | 0.8870 | 0.5812 | **203.1** | **25.63 M** |
| BASE + P3B + KD, residual pruned, floor 0.4 | 0.1089 | 0.8793 | 0.5920 | 203.2 | 25.67 M |
| BASE + P3B + KD, residual pruned | 0.1099 | 0.8780 | 0.5657 | 206.5 | 26.00 M |
| BASE + P3B + KD, largest configuration | 0.1052 | 0.8873 | 0.5674 | 209.8 | 31.90 M |
| BASE + P3B, plain fine-tune | 0.1097 | 0.8770 | 0.5952 | 209.8 | 31.90 M |
| DA3-BASE, fine-tuned (teacher, ×0.1) | 0.0862 | 0.9210 | 0.564 | 729.1 | 101.97 M |

The top four pruned rows span 0.1045 – 0.1099, i.e. **0.0054 — exactly the seed spread measured
in Setup.** They are ordered here, but they are not separated. The one thing that genuinely
distinguishes the best row is not its depth: it is the only configuration under DA3-Small on both
compute and parameters, and its 64-aligned MLP widths make its FP16 engine 21% faster (§6).

Against DA3-Small + KD, the best pruned model is **+0.0101 abs_rel, CI [+0.0086, +0.0116]**,
**+0.207 rmse, CI [+0.170, +0.246]** and **−0.0209 d1, CI [−0.0243, −0.0178]** — a real loss on
depth, with every interval far from zero and the gap itself twice the seed spread, so this one is
not a variance artefact. On pose the difference is **not distinguishable**: +0.0123,
CI [−0.0196, +0.0442], and the same holds at all four thresholds. An earlier version of this
document claimed the pruned model lost on both; at the corrected learning rate that claim does not
survive, and it is withdrawn.

### What actually moved the pruned model

Every run below already carries the corrected importance transform and rank-axis masks; each step
changes one thing, except step 2, which changed two at once and is not attributed further.

| step | change | test abs_rel | Δ |
|---:|---|---:|---:|
| start | P3B on the pretrained release, fine-tuned at backbone lr ×0.1 | 0.1291 | |
| 1 | fine-tune backbone lr ×0.1 → ×1.0 | 0.1186 | **−0.0105** |
| 2 | pruning run also at ×1.0, and KD added | 0.1155 | −0.0031 |
| 3 | importance scale 10 → 1 (the official allocation temperature) | 0.1097 | **−0.0058** |
| 4 | prune the fine-tuned teacher instead of the pretrained release | 0.1066 | −0.0031 |
| 5 | fine-tune backbone lr ×1.0 → ×1.5 | 0.1052 | −0.0014 |

The largest single step is a learning rate. Step 3 is the one change to the *allocation* that
helped: at the official default scale of 10, the allocation is extreme — 13 of 24 sub-blocks
pinned to the ratio floor — and softening it bought 0.0058. Every reallocation tried after that,
below, did not help.

**Which of these steps the variance measurement actually supports.** Only steps 1 (−0.0105) and
3 (−0.0058) exceed the 0.0054 seed spread; steps 2, 4 and 5 (−0.0031, −0.0031, −0.0014) do not,
and individually none of them is established. The cumulative **−0.0239 from 0.1291 to 0.1052 is
far outside the spread** and is real; the attribution to individual steps below 0.005 is not. This
table was built one change at a time before the variance was known, which is the wrong order — the
right order is to measure the noise floor first and only then run an ablation whose steps are
expected to clear it.

The importance defect above is instructive in the same way. Two pruning runs that differed in it —
every block the same width against a depth-dependent profile — and in three other internals (mask
axis, sharpening learning rate, budget actually spent) reached 0.1203 and 0.1206 training-time
validation at epoch 12. The defect was real; its accuracy cost was not measurable.

### What did not work: pruning the residual stream

P3B narrows what is *inside* a block and never touches the residual stream running through the
whole backbone. That stream carries **80% of backbone FLOPs** — the qkv/proj projections (48.1 G)
and the MLP's two matmuls (68.4 G) are all proportional to it, and only attention itself (29.1 G)
is not. Halving it from 768 to 384 frees 58.2 G, which at a fixed budget buys **1.7× more width
inside every block**. I implemented it as one coupled group of 157 tensors spanning patch
embedding, both LayerNorms of every block, all four projection axes, LayerScales and the head's
input.

It measured **worse — 0.1099 against 0.1052 — and that reading has since been withdrawn.** The
0.0047 difference is smaller than the 0.0054 seed spread measured later (Setup), so these two
separately-trained runs do not separate, however tight the bootstrap interval on them was. The
correct statement is that **residual-stream pruning did not help; it is not established that it
hurt.** The hypothesis that the remaining gap was residual-stream capacity still finds no support
either way. (Second caveat, unchanged: FLOPs fell only 1.6% while parameters fell 19%, so this was
never a pure constant-compute reshape.)

Every configuration that followed nevertheless keeps the narrowed stream, for a reason independent
of accuracy: it is what brings the model under DA3-Small on **both** axes, and it halves the
activation traffic that dominates latency on the deployment target (§6).

Three implementation details worth recording, because two of them fail silently:

- The stock DINOv2 attention recovers head width as `C / num_heads` from its input, so a narrowed
  residual makes it compute the wrong head width. It has to be replaced with an attention whose
  inner width is independent of the stream.
- DA3 concatenates local and global tokens before the head, so head-side tensors are indexed at
  **doubled** positions.
- **The camera decoder must be pruned with the stream or the model cannot be evaluated at all.**
  Training never sees this — the trainable subclass drops the camera modules — but every
  evaluation path (`test`, pose evaluation, FLOP counting) builds the full model and runs it. A
  budget-calibration step caught this before a 2.6-hour pruning run would have been wasted.

### What did not work: feeding the starved layer

Profiling DA3-Small against the residual-pruned variant — the one that shares its 384-wide token
stream, so the two are comparable layer for layer — the skeletons turn out **identical**: 12
blocks, width 384, same RoPE start, same cross-view schedule (blocks 5, 7, 9, 11), same token
concatenation. The difference is entirely in how compute is spread across depth:

| block | | DA3-Small | pruned BASE (residual 384) | Δ |
|---:|---|---:|---:|---:|
| 0–4 | local | 8.50 each | 9.8 – 12.0 | **+1.3 … +3.5** |
| 5 | **cross** | 13.61 | 17.76 | +4.15 |
| 7 | **cross** | 13.61 | 16.09 | +2.48 |
| 9 | **cross** | 13.61 | 13.69 | +0.08 |
| 11 | **cross** | 13.61 | **5.36** | **−8.25** |
| | backbone | 123.3 | 132.1 | +8.8 |
| | head | 82.2 | 74.3 | **−7.9** |

**The head is where pruning wins.** Module by module the two heads are now identical —
`features` 64, `out1` 32, same fusion chain, same costs to the second decimal. The whole 7.9
GFLOPs difference is the dead auxiliary branches of §4, which DA3-Small still computes and
discards.

**Block 11 looked like the bottleneck.** It holds 3 of 12 heads and 307 of 3072 MLP neurons —
39% of DA3-Small's compute at that depth — and it is the deepest of the four layers the DualDPT
head reads. A linear depth probe on patch tokens plausibly under-rates exactly such a layer.

**Tested, and refuted.** Raising the keep-ratio floor from 0.10 to 0.40 — an official P3B
hyperparameter, not a change to the method — forces block 11 to at least 5 heads and 1229 MLP
neurons. After calibration it lands at 203.2 GFLOPs and 25.67 M parameters, the first
configuration under DA3-Small on both axes. Against the residual-pruned run it modifies:

| | test | val |
|---|---|---|
| abs_rel | −0.0010 [−0.0021, +0.0001] — not distinguishable | −0.0000 [−0.0012, +0.0011] — not distinguishable |
| d1 | +0.0013 — not distinguishable | −0.0026 [−0.0043, −0.0009] — worse |
| AUC@3° | +0.026 [+0.012, +0.041] — better | −0.010 — not distinguishable |

Nothing on depth. Pose improves on test and not on val, and a difference that changes sign between
splits is not claimed. Block 11 received the least width, but giving it more did not help — read
the other way, BPI's judgement of that layer was right.

A side result: the budget calibrator that chose this configuration had undershot twice in a row
by 1.2–1.6 GFLOPs, both times in the same direction. The cause was a flat per-block attention
cost, whereas the pruning run measures each sub-block and the cross-view blocks cost nearly twice
the local ones (25.7 vs 13.5 GFLOPs). Measuring the vector the same way brought the prediction
error to 0.2 GFLOPs.

### The honest framing

DA3-Small is not a small model trained from scratch; it inherits full DINOv2 ViT-S pretraining.
The pruned model has 25 epochs of pruning plus 25 of fine-tuning to make up for that, and does
not. A claim that pruning beats training a small model would need the small model held to the
same budget — against a pretrained release, the premise favours the baseline from the start.
Worth stating rather than quietly comparing anyway.

The measurements point the same way. Every change that improved the pruned model altered how it
was *trained* — learning rate, initialisation, distillation — or tempered an extreme allocation;
none that reshaped a sane allocation did, whether that meant pruning the residual stream or
feeding the layer BPI had starved. Distillation being worth four times as much here as on
DA3-Small (§3) fits the same reading: the pruned model is short of information, not of structure.
Testing that directly means separating the architecture from the pretraining — done below.

### Testing it directly: the architecture alone, no pretraining

If the gap above is pretraining and not the architecture P3B finds, then stripping pretraining
from both sides should close it. Every weight of both models was re-drawn from DINOv2's own
initialisation (truncated-normal for Linear layers, LayerScale at its configured value) and
verified tensor-by-tensor to differ from the pretrained checkpoint, then trained on KITTI alone —
50 epochs, no KD, backbone lr ×1.0. The comparison uses the residual-pruned, floor-0.4
configuration (§5) rather than the largest pruned model, because it is the one with no size
advantage over DA3-Small.

| | abs_rel ↓ | rmse ↓ | d1 ↑ | AUC@3° ↑ | GFLOPs | params |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained, DA3-Small fine-tuned | 0.0952 | 3.333 | 0.9065 | 0.5824 | 205.6 | 25.93 M |
| Pretrained, pruned (residual 0.5 + floor 0.4) + KD | 0.1089 | 3.577 | 0.8793 | 0.5920 | 203.2 | 25.67 M |
| **Scratch**, DA3-Small architecture | 0.1834 | 5.246 | 0.7172 | 0.1534 | 205.6 | 25.93 M |
| **Scratch**, pruned architecture | 0.1827 | 5.055 | 0.7273 | 0.1671 | 203.2 | 25.67 M |

*(The scratch rows ran 50 epochs with no KD; the pretrained rows are the 25-epoch fine-tune/KD
recipe used everywhere else in this document. The scratch pair is only compared against itself.)*

**The gap disappears when pretraining is removed.** Paired against DA3-Small's architecture with
identical training (test split, paired bootstrap), abs_rel is not distinguishable
(−0.0007 [−0.0042, +0.0030]), and the pruned architecture is *better* on rmse
(−0.191 [−0.283, −0.098]) and d1 (+0.0102 [+0.0022, +0.0180]). AUC@3° is +0.0138 [−0.0072,
+0.0341], not distinguishable. Pretrained, the same pruned configuration loses abs_rel by
+0.0145; from scratch, architecture for architecture, it does not lose at all.

That isolates where the pretrained gap comes from. Pretraining is worth **−0.088 abs_rel** to
DA3-Small (0.1834 → 0.0952) and **−0.072** to the pruned architecture (0.1827 → 0.1089) — DA3-Small
gains about 0.016 more from it, which is close to the whole gap measured with pretraining in
place. P3B finds an architecture that is not worse than DA3-Small's; the pruned model loses
because it inherits a fraction of DA3-BASE's pretraining while DA3-Small keeps all of its own,
not because the pruning method chose badly.

---

## 6 · TensorRT INT8 and Jetson Orin Deployment

Five routes, one shared starting point: the FP32 ONNX of backbone + DualDPT head, static
`(1, 5, 3, 154, 518)`. Every route ends with the **same** `val_trt.py` evaluation used for the
PyTorch rows, so engine numbers are directly comparable to them and to each other.

| Route | Kind | Who decides where quantization happens |
|---|---|---|
| TensorRT FP16 | baseline | — |
| TensorRT implicit INT8 | implicit, `IInt8EntropyCalibrator2` | TensorRT's builder, from calibration it collects itself |
| NVIDIA ModelOpt | explicit Q/DQ | the ONNX graph, entropy calibration |
| AdaRound | explicit Q/DQ + per-block reconstruction | as above, but rounding direction **and activation scales** are learned |
| LS-ViT | explicit Q/DQ + diagonal/low-rank Hessian-weighted reconstruction + QDrop | as AdaRound; the two configs differ in exactly three fields |

**These comparisons are not subject to the training variance of §5.** All five engines are built
from one set of weights, so a difference between routes is a property of quantization, not of a
re-training. That is why the intervals below are quoted without the caveat attached to §5's rows.

### What INT8 costs

**RTX 4090, test split, pruned + KD model at 203.2 G / 25.67 M (PyTorch abs_rel 0.1089)**

| | abs_rel | Δ vs PyTorch [95% CI] | AUC@3° | GPU compute | INT8 compute layers | backbone GEMMs in INT8 |
|---|---:|---|---:|---:|---:|---:|
| PyTorch | 0.1089 | — | 0.592 | — | — | — |
| TensorRT FP16 | 0.1089 | −0.0001 [−0.0001, +0.0000] | 0.593 | 2.749 ms | 0% | 0/24 |
| implicit INT8 | 0.1107 | +0.0017 [+0.0013, +0.0023] | 0.600 | 2.507 ms | 58% | 0/25 |
| ModelOpt INT8 | 0.1103 | +0.0014 [+0.0010, +0.0018] | 0.640 | 2.405 ms | 73% | 11/16 |
| **AdaRound INT8** | 0.1096 | +0.0007 [+0.0005, +0.0009] | **0.595 n.d.** | **2.383 ms** | 75% | 11/13 |
| **LS-ViT INT8** | **0.1092** | **+0.0003 [+0.0001, +0.0005]** | 0.634 | 2.453 ms | 75% | 11/13 |

- **FP16 is free** — indistinguishable from PyTorch on depth and pose — so it, not PyTorch, is the
  baseline every INT8 cost is measured against.
- **Reconstruction earns its keep.** LS-ViT's Hessian-weighted objective leaves only **+0.0003**
  against PyTorch, five times less than ModelOpt's entropy-calibrated PTQ.
- **Implicit INT8 cannot reach the ViT at all.** TensorRT's Myelin pass fuses each Transformer
  block into one region and emits no INT8 GEMM inside it, so INT8 lands only on the head's
  convolutions (53 of 58). **Getting INT8 into the backbone requires explicit Q/DQ** — that is the
  structural reason the explicit routes exist.

### The error is in the activations, not the rounding

Quantizing one side at a time, on val:

| | val abs_rel |
|---|---:|
| FP32 | 0.1096 |
| weights only (W8A32) | 0.1097 |
| **activations only (W32A8)** | **0.1108** |

Weight quantization is nearly free; essentially the whole INT8 penalty is activations. This is
what decides the reconstruction design: **plain AdaRound, which learns only weight rounding, is
indistinguishable from ordinary PTQ here** (−0.0002). Letting the activation scales train with it
is what converts it into a real gain (−0.0005 [−0.0008, −0.0002]). A method that only fixes
rounding is solving the smaller half of this model's problem.

### The latency win came from the pruning stage, and it reversed a conclusion

P3B's bake kept whatever width passed threshold, so MLP hidden widths came out at 1429, 1229 and
similar — **not multiples of 64**. TensorRT therefore sent `fc1` down a generic GEMM. Attention was
never affected, because whole heads are pruned and the width is aligned by construction.

Aligning the widths at the pruning end (largest-remainder rounding, total width preserved):

| | before | after |
|---|---:|---:|
| FP16 engine, GPU compute | 2.761 ms | **2.180 ms (−21%)** |
| GEMMs on the fast `ampere_h16816gemm_256x128` kernel | — | **7 switched over** |

Three builds per side, distributions non-overlapping. **Accuracy cost: none** — a +0.0016 reading
appeared once and sits well inside the 0.0054 seed spread.

**Then the same fix invalidated a ranking I had already written down.** With the widths misaligned,
I had measured explicit INT8 as *faster* than FP16 and reported it. That was an artefact: implicit
and FP16 both run the backbone in FP16 and were both being dragged by 11–12 generic
`xmma_f16f16` GEMMs, while explicit INT8 ran the backbone in INT8 and bypassed the slow path
entirely. Once aligned, FP16 and implicit each gained 21–22% and the explicit routes only 1–4%:

| RTX 4090, after alignment | GPU compute |
|---|---:|
| **implicit INT8** | **1.939 ms** |
| FP16 | 2.180 ms |
| explicit INT8 routes | 2.34 – 2.38 ms |

**On this GPU, explicit INT8 is both slower and less accurate than plain FP16.** The lesson is
methodological rather than technical: the whole comparison had been run on a baseline with a
systematic bias in it, and no amount of statistical care on top of that would have caught it.

### Jetson AGX Orin

Same ONNX, rebuilt on the device (engines are hardware- and TensorRT-version-bound, so only the
graph travels). Seven engines timed in one window, three rounds each, from the best pruned model
(PyTorch abs_rel 0.1045).

| Jetson AGX Orin 64 GB, TRT 10.3 | latency | test abs_rel | rmse | d1 | AUC@3° |
|---|---:|---:|---:|---:|---:|
| **implicit INT8** | **22.60 ms** | 0.1075 | 3.548 | 0.8834 | 0.5630 |
| TensorRT FP16 | 24.29 ms | 0.1045 | 3.493 | 0.8871 | 0.5780 |
| **LS-ViT, head only** | 24.59 ms | **0.1043** | **3.496** | **0.8871** | 0.5594 |
| AdaRound | 24.88 ms | 0.1054 | 3.507 | 0.8858 | **0.5961** |
| LS-ViT | 24.97 ms | 0.1049 | 3.502 | 0.8861 | 0.5734 |
| ModelOpt | 25.13 ms | 0.1057 | 3.497 | 0.8861 | 0.5324 |
| LS-ViT + DLA | 35.11 ms | 0.1047 | 3.500 | 0.8865 | 0.5601 |

- **The two machines agree to 0.0006 on every metric**, using one evaluation protocol across both —
  which is the point of running the project's own evaluator on the device rather than a separate
  deployment harness.
- **The best depth on Orin, 0.1043, is below the PyTorch model it was quantized from (0.1045).**
  Quantizing only the head, with a Hessian-weighted objective, costs nothing measurable.
- **The 4090's ordering reproduces on Orin**: implicit fastest, FP16 next, every explicit INT8 route
  slower than FP16. Offloading to the DLA is 40% slower and was dropped.
- **The real choice is not in the depth column.** Depth spans 0.0032 across all seven engines;
  pose spans **0.064**. Against the FP16 reference (0.5780), AdaRound moves pose **+0.018** while
  ModelOpt moves it **−0.046** — quantization displaces the ray→pose solution, and **the direction
  is not fixed by how good the depth is**: ModelOpt and LS-ViT have near-identical depth and sit at
  opposite ends of the pose column. So the route follows the consumer: **LS-ViT head-only for
  depth, AdaRound if anything downstream uses pose.** On the 4090 checkpoint the displacement went
  the other way for ModelOpt (+0.048), which is the point — it is noise-driven at ±0.02–0.05, not a
  property of the method, and no pose claim between routes is made on one measurement.

### Why explicit INT8 loses, verified rather than assumed

The obvious hypothesis — INT8 kernels are slow on this hardware — is **wrong**, and measuring it
was what redirected the investigation. Timed in isolation on Orin, INT8 GEMMs are **9–31% faster**
than their FP16 counterparts, including 30% faster on the narrowest block.

The actual mechanism is fusion, not arithmetic. Explicit Q/DQ occupies the convolution's
**epilogue slot**, which is where TensorRT would otherwise fuse the activation and the residual
add. Implicit mode emits fused `conv + relu` and `conv + add`; explicit mode emits
`conv + QuantizeLinear` and spills the rest into standalone pointwise layers:

| head region | implicit | explicit |
|---|---:|---:|
| pointwise layers | 19 | **33** |
| time in them | 0.37 ms | **1.01 ms** |

Three candidate fixes were implemented and measured — reverting the offending convolutions to
FP16, putting both residual inputs in one INT8 domain, and pinning the full-resolution head region
— and **all three failed**. The line was closed and documented as closed, because the next person
to look at this (including me) would otherwise pay for the same three experiments again.

### Roughly half of TensorRT 10.10's builds of this model are numerically wrong

Not a tuning issue, and the most expensive thing found in this stage. The same QDQ ONNX with the
same flags occasionally produces an engine that computes a **test abs_rel around 0.37** — a broken
engine, not a degraded one. Measured by rebuilding: ModelOpt's graph gave **3 bad builds out of 3**,
AdaRound's 1 good out of 3.

- The broken builds share a signature (the concat/LayerNorm after patch embedding split into
  separate kernels, a missing fusion in the head, four extra generated layers), and each broken
  engine is **deterministic** — it returns the same wrong answer every run, so this is not a race.
- The same graph is correct under ONNX Runtime, so **the graph is fine and the engine has to be
  validated on its own.**
- Two earlier explanations — `--best`/BF16, then absolute vs relative ONNX paths — each held for a
  while and then produced counterexamples. The path merely perturbed the builder's choices; it was
  never the mechanism. Recording that is the honest version.
- **What shipped is a process, not a diagnosis**: every build is validated on val immediately after
  it is produced, a failure triggers a rebuild, the third attempt falls back to a single auxiliary
  stream, and the outcome is written to a JSON record beside the engine.

A related trap: **one build's latency is a sample, not the route's latency.** Valid builds of the
same route spread 0.03–0.05 ms, while re-timing one engine varies by 0.01 ms — so the spread is the
build, not the measurement. An early single-build comparison had recorded "ModelOpt 2.39 against
AdaRound 2.44" as a route difference. It was not. Every route now builds N engines, validates each,
and the fastest valid one is timed.

### Porting the graph to Orin: three things that had to change

1. **`kernel_shape` must be written on every Conv.** TensorRT 10.3 takes an importer path whenever
   the kernel is a runtime tensor — which a QDQ graph guarantees, since Conv reads `DQ(Q(weight))` —
   and that path does not infer the attribute from the weight. Parsing fails before anything is
   built. Located by extracting each convolution into a minimal model: **our graph fails 56 of 58,
   ModelOpt's passes 58 of 58, and the only difference anywhere in the subgraph is that attribute.**
   It is pure metadata — ONNX Runtime output is bit-identical — and it is now written at export.
2. **The `--best` broken-build problem is specific to TensorRT 10.10.** The same ModelOpt graph
   builds correctly with `--best` on Orin's 10.3, so the workaround above is a 10.10 workaround and
   not an Orin requirement.
3. **Benchmark dependencies must be imported lazily.** The evaluator pulled in the pose-evaluation
   chain at module import; the device has no outbound network, so the script died even when pose
   evaluation was switched off.

### Why INT8 is worth more on Orin than on a 4090

Worth stating because it is the reason the deployment measurement could not be skipped. TensorRT
selects `sm80_xmma_gemm_i8f32` for INT8 — an **Ampere** kernel, a generation behind on Ada, while
Orin is sm87 and runs it natively. Orin's bandwidth is roughly a fifth of the 4090's, and this
head moves 12.76 M-element tensors at full resolution, so halving activation traffic is worth far
more there. **Engines cannot be copied across hardware and neither can conclusions about them.**

---

## 7 · Methods Implemented

| Component | Source | What was built |
|---|---|---|
| **DA3 training path** | Depth Anything 3 | The released forward is wrapped in `@torch.inference_mode()`; a differentiable subclass reimplements it, keeps the head in fp32 so the confidence exponential cannot overflow, and drops the camera modules training never updates |
| **Geometry-aware KD** | this work | Depth L1 + log L1, ray cosine + origin L1, optional point-map and feature terms, shared teacher-confidence mask, scale normalisation shared with the GT loss |
| **P3B** | Pruning by Block Benefit, ICCVW 2025 | 24 sub-blocks; depth-probe BPI replacing the classification aux head; official importance transform, momentum and rank-axis soft masks; water-filling budget in FLOPs; bake into physically smaller modules |
| **Pruned attention** | this work | The stock DINOv2 attention derives head width as `C / num_heads` and cannot express fewer heads at fixed `head_dim` — required because RoPE and QK-norm are tied to `head_dim`, so only whole heads may be removed |
| **DualDPT head pruning** | grouped Fisher, GFP-style | Coupled channel groups across the main and auxiliary fusion chains; dead-branch elimination; shape config saved so a pruned checkpoint reloads without an export step |
| **Residual-stream pruning** | this work | One coupled group of 157 tensors across the whole model, including the camera decoder; a measurement-based budget calibrator, because narrowing the stream invalidates the cost vector the budget is spent against |
| **MLP width alignment** | this work | Largest-remainder rounding of baked MLP widths to a multiple of 64 with the total preserved, so `fc1` reaches TensorRT's fast GEMM kernel; a hard floor mode and a nearest mode, measured against each other |
| **Implicit INT8** | TensorRT | `IInt8EntropyCalibrator2` fed from the project's own calibration windows through device pointers, no second CUDA binding; an explicit optimization profile, required for any network with a dynamic input |
| **Explicit INT8 with reconstruction** | AdaRound (2020), LS-ViT | Per-block reconstruction in PyTorch fake-quant, quantizing at each consumer's input; AdaRound's rounding loss with β annealing, and LS-ViT's diagonal + low-rank Hessian weighting with QDrop, kept one config diff apart from it; a scope switch so the head can be quantized alone |
| **Engine validation and timing** | this work | Best-of-N builds with every engine validated on val before it can be selected, rebuild on failure, fallback on the third attempt, and a JSON record per build; latency measured after the fact on an idle card, medians over rounds, all routes in one window |
| **Jetson Orin deployment** | this work | `kernel_shape` written at export for TensorRT 10.3's importer path; lazily imported benchmark dependencies for a network-isolated device; the project's own evaluator run on-device so numbers compare directly to the 4090's |

---

## Caveats

- KITTI odometry, not the Eigen split — the multi-view window and pose evaluation need
  consecutive frames with ground-truth trajectories.
- LiDAR supervises 4.4% of pixels; depth metrics are sparse-GT metrics with per-frame median
  alignment, not comparable to dense-GT benchmarks.
- Following an existing implementation choice, prediction and GT are both divided by the same
  GT-derived scale, whereas the paper normalises GT only. Metrics are unaffected (median
  alignment, scale-free pose) but "reproducing DA3" carries this caveat.
- The teacher row is still on the ×0.1 recipe (§2), so the teacher–student gap is not a
  matched-recipe number.
- The pruning stage diverges from the official implementation in two documented ways: masks
  update once per epoch rather than three times, and the probe uses the patch branch only,
  because DA3's class-token slot carries a camera token rather than a label.
- Pose AUC has a bootstrap width of roughly ±0.03 on this split — wider than any difference
  measured between compressed variants, so no pose claim is made between them.
- Run-to-run variance for **pruned** runs is 0.0054 abs_rel, measured from one seed pair. A single
  pair is a weak estimate of a spread; it is quoted because it is measured rather than assumed, and
  it is already large enough to retire the two claims noted in §3 and §5. Differences between
  pruned configurations below ≈0.005 in this document are ordered but not separated.
- The quantization tables are on two different checkpoints: the 4090 accuracy table on the
  floor-0.4 model (PyTorch 0.1089), the Orin table on the 64-aligned model (PyTorch 0.1045). They
  are not rows of one table and are not compared across.
- Roughly half of TensorRT 10.10's builds of this graph are numerically broken (§6). The mitigation
  is validation and rebuilding, not a root cause — the cause is not identified, and "the version
  that can be deployed" is the honest scope of every 10.10 latency number here.
- LS-ViT's latency on the 4090 is reported from graphs whose MLP widths are aligned; ModelOpt's
  graph did not produce a valid build under padding, so that particular comparison covers only the
  configurations that build.

---

## About

I work on making vision models small enough and fast enough to run on embedded hardware —
structured pruning (CNN channels and Transformer blocks), knowledge distillation, PTQ with
Hessian-aware reconstruction, and TensorRT INT8 deployment on NVIDIA Jetson. This project extends
that work from classification to multi-view metric depth, where the output is geometry, the
evaluation has to be statistical rather than a single accuracy number, and the last stage is an
engine running on the target device rather than a checkpoint.

**Contact:** [LinkedIn](https://linkedin.com/in/tommychen19920812) · a0912986300@gmail.com
