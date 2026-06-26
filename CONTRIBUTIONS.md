# Contributions

This repository extracts and builds upon the main work from our team JEPAdormi as part of the Hack the World(s) hackathon,
which was initially pushed to the fork [`Jules02/eb_jepa`](https://github.com/Jules02/eb_jepa).

- **Fork point (last upstream commit):** `966e61e` — *"Upd README, gitignore, and assets"* (2026-02-04)
- **Our hackathon work:** 34 non-merge commits (19–20 June 2026) by
  *aduplessi vivatech*, *Adnan Ben Mansour*, *Jules02 / jdupont vivatech*.

> **Scope of this repository.** This repo extracts and builds upon our work on 
> **Gray-Scott reaction-diffusion systems** only. During the hackathon we
> also produced some early work on maze exploration and distance-landscape
> visualization; it is intentionally **not** included here but remains in the original fork's history.
> The commit trace in §3 lists our hackathon work for the record.

Everything below is what we added or changed.

---

## 1. New code we authored

### `gray_scott/`
Temporal Gray-Scott reaction-diffusion VJEPA, end to end:
- **Model & training** — `main.py` (encoder/predictor/JEPA build), `train_decoder.py`.
- **Evaluation** — `eval.py`, `eval_common.py`, `eval_regimes.py`, `eval_compare.py`,
  `eval_baselines.py`, `unroll_ood.py`, `final_table.py` — VRMSE (per-channel u/v,
  pooled-ratio, mean-of-ratios variants) + OOD unrolling + per-regime reporting.
- **Baselines** — `baselines.py`, `_well_baselines.py` (ResUNet, FNO/TFNO,
  UNetClassic, UNetConvNext via [The Well](https://github.com/PolymathicAI/the_well)).
- **Latent analysis** — `pca.py`, `pca_entropy_anim.py`, `plot_pca*.py`,
  `latent_potential.py`, `latent_walk.py`, `probe.py`, `field_metrics.py`.
- **Visualization** — `visualize.py`, `viz_rollouts.py`, `viz_regimes_gif.py`,
  `perturb_gif.py`, `perturb_ab_gif.py`, `spirals_2panel.py`, `gif_pca.py`,
  `gif_slides.py`, `render_slides.py`, `slides_metrics.py`, plotting helpers.
- **Configs** — `cfgs/{train,train_large,train_stride1,train_vjepa,train_vjepa_v2,eval}.yaml`.
- **Docs** — `README.md`, `DESIGN.md`, `DESIGN_large.md`.

### Root-level launchers we wrote
`train_gs_vjepa.sh`, `train_gs_vjepa_v2.sh`, `train_decoder.sh`,
`slurm_*.sh` (analysis, pca, perturb, probe, viz, eval),
`run_*.sh` (baselines, field, final_table, pca, plots, potential, render_slides,
viz, walk), `vrmse.sh`.

---

## 2. Upstream core files we modified

(Vendored under `eb_jepa/`.)

| File | +/− | What we changed |
|------|-----|-----------------|
| `eb_jepa/losses.py` | +142 / −5 | Added regularizer terms (signature / temporal-distance) and related loss code. |
| `eb_jepa/architectures.py` | +31 / −18 | GroupNorm support. |
| `eb_jepa/jepa.py` | +2 / −2 | Minor wiring. |

We also added the `eb_jepa/datasets/gray_scott/` package (dataset loader for the
Gray-Scott data from [The Well](https://github.com/PolymathicAI/the_well)).

> Our hackathon also modified `eb_jepa/planning.py`, `state_decoder.py`, and
> `datasets/utils.py`, but those changes served the `ac_video_jepa`/maze track and
> are not part of this Gray-Scott extraction.

---

## 3. Commit trace (our team, chronological)

<!-- generated from: git log 966e61e..HEAD --no-merges --author=... -->
- 2026-06-19 reservation
- 2026-06-19 reserv
- 2026-06-19 viz
- 2026-06-19 fix viz
- 2026-06-19 fix viz 2
- 2026-06-19 ajout de sigreg pour ac_video
- 2026-06-20 ac_video_jepa: temporal-distance regularizer term + maze train configs/scripts
- 2026-06-20 track4: implement gray-scott temporal JEPA (encoder, predictor, decoder, VRMSE eval)
- 2026-06-20 declare cluster reservation name in smoke test script
- 2026-06-20 track4: implement gray-scott temporal JEPA (encoder, predictor, decoder, VRMSE eval)
- 2026-06-20 viz_distance_landscape: support maze in addition to two_rooms
- 2026-06-20 eval: fix VRMSE to be per-channel (u,v) then averaged; add Table 3 windows [6:12]/[13:30]
- 2026-06-20 eval: switch to mean-of-ratios VRMSE; add per-channel u/v diagnostics
- 2026-06-20 eval: revert to pooled-ratio per channel (mean-of-ratios blew up on near-uniform v frames)
- 2026-06-20 add visualization, decoder trainer, and small model reconstruction plot
- 2026-06-20 add v-channel first/last frame visualization (reveals zero-pattern trajectories)
- 2026-06-20 Add Gray-Scott JEPA visualizations and paper-metric eval
- 2026-06-20 viz: longer trajectories (H=60, 2400s) at 10fps + add pooled-VRMSE eval scripts
- 2026-06-20 viz: add 4-panel comparison GIFs (Truth | JEPA | UNetClassic | CNextU-Net)
- 2026-06-20 viz: 5-panel all-models comparison GIFs (Truth|JEPA-small|JEPA-large|UNet|CNext)
- 2026-06-20 viz: stride=1 fair comparison + stride=4 all-models with large ep5
- 2026-06-20 add GroupNorm support + vjepa_v2 comparison GIFs
- 2026-06-20 Add per-regime eval + visualization; gitignore slurm logs
- 2026-06-20 Add stride-4 baselines (ResUNet + FNO) to comparison GIFs
- 2026-06-20 gray_scott: factor duplicated the_well stub boilerplate into shared helper
- 2026-06-20 Add latent analysis: PCA, perturbation GIFs (A+delta vs B+delta)
- 2026-06-20 Redo perturbation GIFs with 2-row A/B split layout
- 2026-06-20 Fix perturb_ab GIF layout: taller figure, clear A/B row labels
- 2026-06-20 gray_scott: add OOD unrolling eval + pooled-VRMSE regime reporting
- 2026-06-20 gray_scott: stop tracking generated viz binaries (208 MB)
- 2026-06-20 gray_scott: document the script suite + VRMSE variants in README
- 2026-06-20 gray_scott: render regime GIF as green/red R=A,G=B composite
- 2026-06-20 Add 2-panel spirals A-channel perturbation GIF
- 2026-06-20 gray_scott: shrink regime GIF + group panels by dynamic/static
- 2026-06-20 gray_scott + ac_video: latent analysis, baselines, field metrics, slides
