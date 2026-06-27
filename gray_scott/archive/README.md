# `gray_scott/archive/` — analysis & visualization scripts

These are the exploratory analysis and figure/GIF-producing scripts from the
hackathon. They produced the plots and animations in our talk but are **not part
of the core train/eval pipeline**, so they live here to keep `gray_scott/` tidy.

They are still fully runnable — just from the `archive` sub-package:

```bash
python -m gray_scott.archive.visualize --ckpt <ckpt> --H 60
python -m gray_scott.archive.pca       --ckpt <ckpt>
python -m gray_scott.archive.unroll_ood --ckpt <ckpt> --F 0.020 --k 0.0515
```

They import the core modules (`gray_scott.eval`, `gray_scott.eval_common`,
`gray_scott.baselines`, …) from the parent package, so the repo must be on
`PYTHONPATH` (run from the repo root, as the `scripts/` launchers do).

## What's here

| Area | Scripts |
|------|---------|
| Rollout / field GIFs | `visualize.py`, `viz_rollouts.py`, `viz_regimes_gif.py`, `spirals_2panel.py` |
| Perturbation studies | `perturb_gif.py`, `perturb_ab_gif.py` |
| Latent analysis | `pca.py`, `plot_pca.py`, `plot_pca_entropy.py`, `pca_entropy_anim.py`, `latent_walk.py`, `latent_potential.py`, `probe.py` |
| Metrics & tables | `field_metrics.py`, `final_table.py`, `plot_results.py`, `slides_metrics.py`, `replot_bars.py`, `replot_field.py` |
| Slides / OOD | `render_slides.py`, `gif_slides.py`, `gif_pca.py`, `analysis.py`, `unroll_ood.py` |

The matching `scripts/run_*.sh` and `scripts/slurm_*.sh` launchers already invoke
these under their new `gray_scott.archive.*` paths.
