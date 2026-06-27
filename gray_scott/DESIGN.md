# Gray-Scott temporal V-JEPA — design notes

## Research question

Can a JEPA learn the *dynamics* of a PDE by predicting future *latents* (not
pixels)? And how does latent-space prediction compare — in field-space VRMSE — to
a persistence baseline and to neural-operator surrogates (FNO / U-Net)?

## Data

`polymathic-ai/gray_scott_reaction_diffusion` ([The Well](https://github.com/PolymathicAI/the_well),
Ohana et al. 2024). Two chemical fields **A**/**B** diffuse and react on a
128×128 grid across 6 visually distinct regimes (spots, worms, maze, spirals,
gliders, bubbles), 1001 timesteps per trajectory. Each training item is a clip of
`n_frames` sampled at stride 4; fields are z-scored per channel; train/valid/test
splits are trajectory-disjoint (no leakage).

## Model: predictive (temporal) JEPA

```
context frames  x[:, :, :2]
   │ [encoder]  ResNet5(in_d=2, h=henc, out_d=D)   stride-1, full 128×128
   ▼
z[:, :, :2] ──[predictor]──► ẑ[:, :, 2:]   predicted future latents
              StateOnlyPredictor(ResUNet(in_d=2D, h=hpre, out_d=D), context_length=2)
   │
z[:, :, 2:] ──[target]────►  ground-truth future latents (same online encoder)

Loss = SquareLossSeq(ẑ, z_target)            # prediction (plain L2)
     + VCLoss(z, proj=Projector(D→4D→4D))    # anti-collapse (VICReg-style)
```

This is a **predictive** JEPA, not a two-view contrastive one: the predictor rolls
latents forward in time (no pixel reconstruction), and the "target" is the *online*
encoder applied to ground-truth future frames — there is no separate EMA network,
so `VCLoss` is what prevents representational collapse.

### Why these components

- **Encoder `ResNet5`, all strides=1, no pooling.** Must preserve 128×128 spatial
  structure so a light decoder can map latents back to fields for VRMSE.
  `ImpalaEncoder` was rejected — it flattens to `[B,D,1,1]`, making field decoding
  impossible. `TemporalBatchMixin` folds `T` into the batch dim for 5D inputs.
- **Predictor `StateOnlyPredictor` + `ResUNet`.** A U-Net maps latent *fields*
  forward in time: skip connections keep spatial detail while the bottleneck
  captures global dynamics. It takes the 2 most recent latents (channel-concat,
  `in_d=2D`) and predicts the next; `context_length=2` matches the unroll. An RNN
  predictor was rejected because it collapses the spatial layout the decoder needs.
- **Anti-collapse `VCLoss`** (`std_coeff=10`, `cov_coeff=100`, projector D→4D→4D):
  the variance term keeps each latent dim informative; the covariance term
  decorrelates dims. Projecting before the penalty (VICReg convention) avoids
  distorting the raw latents.
- **Prediction loss `SquareLossSeq`** (plain MSE): latents already live in a
  meaningful metric space (z-scored inputs, stride-1 encoder), so L2 suffices.
- **Decoder (eval only):** `Conv2d(D,64,3) → GELU → Conv2d(64,64,3) → GELU →
  Conv2d(64,2,1)`. No upsampling needed (stride-1 encoder ⇒ latent already 128×128).
  Trained separately with the JEPA frozen; its reconstruction error is the
  **irreducible floor** `VRMSE_floor = VRMSE(decode(encode(true)), true)` — the best
  any predictor on this encoder can do.

## VRMSE metric

`VRMSE = sqrt( Σ_space (pred − true)² / Σ_space (true − μ_true)² )`, with numerators
and denominators **aggregated across the whole eval set before dividing**. This
matters: per-sample ratios blow up on near-uniform frames (channel B has near-zero
spatial variance in some regimes); the aggregated ratio stays stable. Three curves
are reported per horizon `h=1..H`:

| Curve | Definition |
|---|---|
| `jepa` | `decode(rollout_latents(context))` |
| `persistence` | repeat last context frame |
| `floor` | `decode(encode(ground-truth frame))` |

A healthy model satisfies `floor < jepa < persistence` at all horizons, with JEPA
near the floor at short horizons.

## Training & scaling: dev → large

The first **dev** run (`gray_scott_dev`, D=16) exposed the key bottleneck — the
**encoder/decoder floor was too high to compete**: floor VRMSE ≈ 0.54 at h=1
(decode∘encode already loses ~54% of variance), versus an FNO target of ~0.1365.
JEPA still beat persistence at h≥3 (its long-horizon advantage), but no predictor
can beat a 0.54 floor. The **large** run addresses this, mostly by giving the
encoder/decoder capacity:

| Hyperparameter | dev | large | Rationale |
|---|---|---|---|
| Latent dim `D` | 16 | 64 | 4× more latent channels ⇒ far lower reconstruction floor |
| Encoder hidden `henc` | 32 | 128 | more capacity for spatially coherent features |
| Predictor hidden `hpre` | 32 | 64 | dynamics modelling only; D=64 already carries the signal |
| `n_frames` | 10 | 16 | matches spec; 10 predicted frames/clip vs 4 (2.5× more signal) |
| JEPA steps | 4 | 10 | 2 context + 10 predicted = 12 ≤ 16 frames used |
| Batch size | 8 | 32 | dev used <5% of GPU memory |
| `epoch_size` | 8000 | 28800 | full dataset per epoch (dev saw only ~28%) |
| Optimizer | Adam 1e-3 | Adam 3e-4 | lower stable LR for the larger model |
| Epochs | 20 | 50 | more params need more epochs; VRMSE eval every 5 |

**OOM lesson:** the first scaled attempt used `hpre=256, batch=64`. Because the
ResUNet runs at full 128×128 with all frames folded into the batch dim (effective
batch 64×10=640), this hit 171/184 GB and crashed. The fix was a *lighter*
predictor (`hpre=64`), not a heavier one — the predictor only needs to learn the
temporal transition, while the D=64 latent carries the spatial detail.

With D=64 / henc=128 the reconstruction floor is expected well below 0.30;
whether the predictor then approaches that floor at short horizons (competing with
FNO's ~0.1365) is what training decides.

Checkpoints: `latest.pth.tar` every epoch, `epoch_N.pth.tar` every 5. W&B
project `eb_jepa`. Configs in [`cfgs/`](cfgs/) (`train.yaml` = dev,
`train_large.yaml` = large).
