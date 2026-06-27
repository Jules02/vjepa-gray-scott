# V-JEPA for Gray-Scott dynamics

**Video Joint Embedding Predictive Architecture (V-JEPA)** for spatiotemporal PDE dynamics —
the [Gray-Scott reaction-diffusion system](https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/).

<p align="center">
  <img src="concentration_A_normalized.gif" alt="Gray-Scott reaction-diffusion field (normalized concentration of species A) evolving over time" width="400">
</p>

This repository is a **focused extraction** of the work our team JEPAdormi produced during the
24-hour [Hack the World(s) hackathon](https://www.hacktheworlds.fr) (we won first prize! 🏆).
Rather than ship our full fork, we kept only the Gray-Scott track we worked on,
plus the minimal upstream library code required for it to run. It also serves as a
**starting point** for further work on V-JEPA for PDE prediction.

## Presentation

Our hackathon presentation is included as **[`HTW_JEPAdormi.pdf`](HTW_JEPAdormi.pdf)**.
View the live version
**[here](https://docs.google.com/presentation/d/1So5CY2_ktrwkIdeoWEdlRiKWU-_UbJu_tTGfFIrUHR4/edit?usp=sharing)**
to see the GIFs in motion.

## How it works

We train a temporal JEPA to predict the **latent** dynamics of the Gray-Scott
field, then decode latents back to (u, v) frames and score multi-step rollouts
with VRMSE against persistence and against The Well's neural baselines
(ResUNet, FNO/TFNO, U-Net, ConvNeXt-U-Net). See
**[`gray_scott/README.md`](gray_scott/README.md)** for the model design, the data,
and the full script suite, and **[`gray_scott/DESIGN.md`](gray_scott/DESIGN.md)**
for architecture notes.

## Setup

```bash
# Python 3.12 (see .python-version)
pip install -e .          # installs the vendored eb_jepa core + deps
pip install the-well      # required for the Gray-Scott data + baseline models
```

**Data.** The Gray-Scott trajectories come from
[The Well](https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/).
The dataset location is set by `ROOT` in
`eb_jepa/datasets/gray_scott/dataset.py` — point it at your local copy (it
currently holds the cluster path used during the hackathon). See the
[track README's Data section](gray_scott/README.md#data) for details.

> **Note:** `eb_jepa` imports `scikit-learn`, which must match your installed
> NumPy. On a NumPy 1.x/2.x ABI error from inside sklearn, pin a matching pair
> (e.g. `numpy<2` with an older sklearn, or an sklearn wheel built for NumPy 2).

## Running

```bash
# Train
python -m gray_scott.main --fname gray_scott/cfgs/train.yaml

# Evaluate a checkpoint (VRMSE per horizon vs persistence)
python -m gray_scott.eval --ckpt <run>/latest.pth.tar --H 30

# Train + score the neural baselines
python -m gray_scott.baselines --split test --H 30
```

The `train_*.sh`, `slurm_*.sh` and `run_*.sh` files in [`scripts/`](scripts/) are the
SLURM / convenience launchers we used on the cluster (submit them from the repo
root, e.g. `sbatch scripts/train_gs_vjepa.sh`); the full catalogue of
analysis and visualization scripts is documented in
[`gray_scott/README.md`](gray_scott/README.md#scripts).

## More

- **[CONTRIBUTIONS.md](CONTRIBUTIONS.md)** — what is ours vs. vendored from
  upstream, the fork point, and the commit trace.
- **[`gray_scott/README.md`](gray_scott/README.md)** — the track in depth: data,
  model, scripts, and the VRMSE metric variants.

## Credits

Built on the hackathon organizer's [`eb_jepa`](https://github.com/Trick5t3r/eb_jepa),
itself built on FAIR's [`eb_jepa`](https://github.com/facebookresearch/eb_jepa).

## License

[Apache License 2.0](LICENSE) — the same license as the upstream `eb_jepa` we
build on. The vendored `eb_jepa/` code remains under its original Apache 2.0
license; our additions are also released under Apache 2.0. Attribution details
are in [NOTICE](NOTICE).
