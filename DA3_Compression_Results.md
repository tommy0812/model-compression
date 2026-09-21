# Depth Anything 3 Compression — Results

**DA3 (DINOv2 + DualDPT) · KITTI odometry · Geometry-aware knowledge distillation + P3B block pruning**

*By [Tommy Chen (陳龍懷)](https://linkedin.com/in/tommychen19920812) — Senior AI Algorithm
Engineer, Compal Electronics. M.S. Electrical and Computer Engineering, NYCU.*

Compressing a multi-view metric-depth model instead of a classifier. DA3 predicts dense
depth, ray maps and point maps from a 5-frame window, so every compression decision has to
be judged on geometry, not on Top-1. Every number below comes from a full-split evaluation
run or a CUDA FLOP count — never from a training-time proxy — and differences carry paired
bootstrap confidence intervals, so "better" means measurably better.

---

## Status

| Stage | State |
|---|---|
| Reproduce DA3 fine-tuning on KITTI | **Done** — one recipe, used unchanged by every row |
| DA3-BASE teacher, DA3-Small student, 4 distillation variants | **Done** — §2, §3 |
| Compute-parity analysis and DualDPT head pruning | **Done** — §4 |
| P3B pruning of DA3-BASE to DA3-Small's compute | **Done on compute, lost on accuracy** — §5 |
| Closing the accuracy gap: 4 independent levers | **3 measured, 1 running** — §5 |

The headline result is negative: at matched compute, a pruned DA3-BASE does not reach a
DA3-Small that carries full DINOv2 pretraining. What follows is the evidence that this is a
property of the comparison rather than of the implementation — including two defects found
in my own port, and one in my own baseline, that had to be fixed before the claim was worth
making.

---

## TL;DR

| | |
|---|---|
| **Compute parity reached by pruning** | 729.1 → **204.9–209.8 GFLOPs** against DA3-Small's 205.6; 101.97 M → **26.0–31.9 M** params |
| **Free compute found by profiling** | **31.5 GFLOPs and 1.16 M params** removed bit-exactly — the DualDPT head computes three auxiliary pyramid branches whose output is never read. DA3-Small still pays 7.9 GFLOPs for them |
| **Port defect found and fixed** | P3B's block importance had collapsed to `0.693 / param_count`, silencing the method's central signal. Restoring the official max-rescaling recovered a **4-orders-of-magnitude** importance spread |
| **Baseline defect found and fixed** | A backbone learning-rate multiplier of ×0.1, inherited from an earlier recipe, was throttling **every row**. Releasing it improved DA3-Small by **0.0062 abs_rel — 6× the total effect of all four distillation variants** |
| **Distillation is worth far more after pruning** | On DA3-Small, KD buys **−0.0011** abs_rel. On the pruned model it buys **−0.0045** — **4×** — because pruning destroys information that a teacher can restore, while a pretrained release has little left to recover |
| **Best pruned result** | **0.1052** abs_rel at 209.8 GFLOPs, against DA3-Small's **0.0944**. Gap **+0.0100**, CI [+0.0082, +0.0118] — still a real loss, from 0.1291 at the start |
| **Where the remaining gap lives** | Measured, not guessed: the two models have *identical* skeletons, and the pruned one starves **block 11 to 3/12 heads and 307/3072 MLP** — the deepest of the four layers the DPT head reads |

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
   25.93 M / 205.6 GFLOPs                   31.90 M / 209.8 GFLOPs
   0.0944 abs_rel                           0.1052 abs_rel
```

Both routes end at the same compute; §5 is about why they do not end at the same accuracy.

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
| Significance | Paired bootstrap over windows, 10,000 resamples. **All method decisions taken on val only.** Seed-to-seed noise measured at ±0.0015 abs_rel |
| Compute accounting | CUDA `FlopCounterMode` on the inference path. A CPU count silently misses every `scaled_dot_product_attention` — exactly 42.6 GFLOPs for DA3-Small |
| Parameter accounting | backbone + head, i.e. the depth/ray path actually executed. Camera encoder/decoder excluded: training drops them and pose comes from the ray head |
| Hardware | 2 × RTX 4090 (24 GB). DA3-BASE fine-tuning peaks at 23.9 GB per card |

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
| **DA3-Small, fine-tuned (×1.0)** | **0.0952** | 3.293 | 0.9065 | 0.5824 |
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
left to recover. **This is the result that justifies distillation as a post-pruning step rather
than an alternative to it.** (AUC differences of this size sit inside the ±0.03 bootstrap width
for pose and are not claimed as real.)

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
| **BASE + P3B + KD** (best) | **0.1052** | 0.8873 | 0.5674 | 209.8 | 31.90 M |
| BASE + P3B, plain fine-tune | 0.1097 | 0.8770 | 0.5952 | 209.8 | 31.90 M |
| BASE + P3B + KD, residual pruned | 0.1099 | 0.8780 | 0.5657 | 206.5 | 26.00 M |
| DA3-BASE, fine-tuned (teacher, ×0.1) | 0.0862 | 0.9210 | 0.564 | 729.1 | 101.97 M |

Against DA3-Small fine-tuned, the best pruned model is **+0.0100 abs_rel, CI [+0.0082, +0.0118]**
and **−0.0192 d1, CI [−0.0232, −0.0155]** — a real loss on depth, with the interval far from
zero. On pose the difference is **not distinguishable**: +0.0068, CI [−0.0242, +0.0372]. An
earlier version of this document claimed the pruned model lost on both; at the corrected learning
rate that claim does not survive, and it is withdrawn.

The pruned line moved **0.1291 → 0.1186 → 0.1155 → 0.1097 → 0.1066 → 0.1052** across four
independent levers — corrected block importance, rank-axis masks, pruning from the fine-tuned
teacher instead of the pretrained release, and a released backbone learning rate. It closed
roughly half the gap and then stopped.

### What did not work: pruning the residual stream

P3B narrows what is *inside* a block and never touches the residual stream running through the
whole backbone. That stream carries **80% of backbone FLOPs** — the qkv/proj projections (48.1 G)
and the MLP's two matmuls (68.4 G) are all proportional to it, and only attention itself (29.1 G)
is not. Halving it from 768 to 384 frees 58.2 G, which at a fixed budget buys **1.7× more width
inside every block**. I implemented it as one coupled group of 157 tensors spanning patch
embedding, both LayerNorms of every block, all four projection axes, LayerScales and the head's
input.

It made the model **worse**: 0.1099 against 0.1052. The hypothesis that the remaining gap was
residual-stream capacity is not supported. (Honest caveat: FLOPs fell only 1.6% but parameters
fell 19%, so this was not a pure constant-compute reshape and part of the regression is simply a
smaller model.)

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

### Where the remaining gap actually lives

Profiling DA3-Small against the residual-pruned variant — the one that shares its 384-wide
token stream, so the two are directly comparable layer for layer — the skeletons turn out
**identical**: 12 blocks, width 384, same RoPE start, same cross-view schedule (blocks 5, 7, 9,
11), same token concatenation. No topological difference explains the gap. The difference is
entirely in how compute is distributed across depth:

| block | | DA3-Small | pruned BASE (residual 384) | Δ |
|---:|---|---:|---:|---:|
| 0–4 | local | 8.50 each | 9.8 – 12.0 | **+1.3 … +3.5** |
| 5 | **cross** | 13.61 | 17.76 | +4.15 |
| 7 | **cross** | 13.61 | 16.09 | +2.48 |
| 9 | **cross** | 13.61 | 13.69 | +0.08 |
| 11 | **cross** | 13.61 | **5.36** | **−8.25** |
| | backbone | 123.3 | 132.1 | +8.8 |
| | head | 82.2 | 74.3 | **−7.9** |

Two things fall out of this table.

**The head is where pruning wins.** Module by module the two heads are now identical — `features`
64, `out1` 32, same fusion chain, same costs to the second decimal. The entire 7.9 GFLOPs
difference is the dead auxiliary branches of §4, which DA3-Small still computes and discards.

**Block 11 is starved.** It holds 3 of 12 attention heads and 307 of 3072 MLP neurons — 39% of
DA3-Small's compute at the same depth — and it is the *deepest of the four layers the DualDPT
head reads*. Across those four layers the pruned model spends 17.8 / 16.1 / 13.7 / 5.4 against a
flat 13.6, i.e. it front-loads the head's own input budget and drains the deepest layer.

This is a predictable consequence of the probe. A linear depth readout on patch tokens converges
early on shallow blocks and reports near-zero benefit for late ones, so water-filling drains
exactly the layers the real head consumes. At the first allocation 17 of 24 sub-blocks had
negative Δψ, the most negative being the last block; importance momentum and per-epoch
re-measurement recover most of that within four epochs, but blocks 9 and 11 never earn budget
back.

**The test now running** raises the keep-ratio floor from 0.10 to 0.40 — an official P3B
hyperparameter, not a modification of the method — which lifts block 11 to 5 heads and 1229 MLP
neurons. The calibrated configuration lands at **204.9 GFLOPs and 26.02 M parameters, under
DA3-Small on both axes for the first time**. A targeted variant that pins only the head's four
input layers to DA3-Small's exact width is implemented and held in reserve; it would be a
deviation from the reference implementation and labelled as such.

### The honest framing

DA3-Small is not a small model trained from scratch; it inherits full DINOv2 ViT-S pretraining.
The pruned model has 25 epochs of pruning plus 25 of fine-tuning to make up for that, and does
not. A claim that pruning beats training a small model would need the small model held to the
same budget — against a pretrained release, the premise favours the baseline from the start.
Worth stating rather than quietly comparing anyway.

---

## 6 · Methods Implemented

| Component | Source | What was built |
|---|---|---|
| **DA3 training path** | Depth Anything 3 | The released forward is wrapped in `@torch.inference_mode()`; a differentiable subclass reimplements it, keeps the head in fp32 so the confidence exponential cannot overflow, and drops the camera modules training never updates |
| **Geometry-aware KD** | this work | Depth L1 + log L1, ray cosine + origin L1, optional point-map and feature terms, shared teacher-confidence mask, scale normalisation shared with the GT loss |
| **P3B** | Pruning by Block Benefit, ICCVW 2025 | 24 sub-blocks; depth-probe BPI replacing the classification aux head; official importance transform, momentum and rank-axis soft masks; water-filling budget in FLOPs; bake into physically smaller modules |
| **Pruned attention** | this work | The stock DINOv2 attention derives head width as `C / num_heads` and cannot express fewer heads at fixed `head_dim` — required because RoPE and QK-norm are tied to `head_dim`, so only whole heads may be removed |
| **DualDPT head pruning** | grouped Fisher, GFP-style | Coupled channel groups across the main and auxiliary fusion chains; dead-branch elimination; shape config saved so a pruned checkpoint reloads without an export step |
| **Residual-stream pruning** | this work | One coupled group of 157 tensors across the whole model, including the camera decoder; a measurement-based budget calibrator, because narrowing the stream invalidates the cost vector the budget is spent against |

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

---

## About

I work on making vision models small enough and fast enough to run on embedded hardware —
structured pruning (CNN channels and Transformer blocks), knowledge distillation, PTQ/QAT with
Hessian-aware reconstruction, and TensorRT INT8 deployment on NVIDIA Jetson. This project extends
that work from classification to multi-view metric depth, where the output is geometry and the
evaluation has to be statistical rather than a single accuracy number.

**Contact:** [LinkedIn](https://linkedin.com/in/tommychen19920812) · a0912986300@gmail.com
