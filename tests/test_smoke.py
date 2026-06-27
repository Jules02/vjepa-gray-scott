"""Smoke tests — guard against dependency / refactor breakage.

These need only the core install (`pip install -e .`); no dataset, GPU, or The
Well. They verify that the core modules import, the model assembles, and the
dataset path resolves from the environment.
"""
import importlib

import pytest

# Core train + eval path — must import with only the core dependencies.
CORE_MODULES = [
    "eb_jepa.logging",
    "eb_jepa.nn_utils",
    "eb_jepa.architectures",
    "eb_jepa.losses",
    "eb_jepa.jepa",
    "eb_jepa.training_utils",
    "eb_jepa.datasets.gray_scott.dataset",
    "gray_scott.main",
    "gray_scott.eval",
    "gray_scott.eval_common",
    "gray_scott.eval_regimes",
    "gray_scott.train_decoder",
    "gray_scott.baselines",
    "gray_scott._well_baselines",
]

# These import The Well at module load -> only when the [baselines] extra is in.
THE_WELL_MODULES = [
    "gray_scott.eval_compare",
    "gray_scott.eval_baselines",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", THE_WELL_MODULES)
def test_the_well_imports(module):
    pytest.importorskip("the_well")
    importlib.import_module(module)


def test_model_builds():
    """Assemble the temporal-JEPA from a tiny config (no data, no GPU)."""
    from omegaconf import OmegaConf

    from gray_scott.main import build_encoder, build_jepa

    cfg = OmegaConf.create(
        dict(dobs=2, henc=16, dstc=8, hpre=8, std_coeff=1.0, cov_coeff=1.0, norm="group")
    )
    jepa = build_jepa(build_encoder(cfg), cfg)
    assert sum(p.numel() for p in jepa.parameters()) > 0


def test_dataset_root_resolves_from_env(monkeypatch):
    """The de-hardcoded data path honours $GRAY_SCOTT_DATA_ROOT."""
    monkeypatch.setenv("GRAY_SCOTT_DATA_ROOT", "/tmp/gs_test_root")
    import eb_jepa.datasets.gray_scott.dataset as ds

    importlib.reload(ds)
    assert ds.ROOT == "/tmp/gs_test_root"
