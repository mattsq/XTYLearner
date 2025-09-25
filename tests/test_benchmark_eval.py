import torch

from eval import BenchmarkDataBundle, ModelBenchmarker


def test_benchmarker_builds_ganite_optimizers():
    benchmarker = ModelBenchmarker()
    bundle = BenchmarkDataBundle(
        train_loader=None,
        val_loader=None,
        x_dim=2,
        y_dim=1,
    )

    model, optimizers = benchmarker._build_model_components("ganite", bundle)

    assert isinstance(optimizers, tuple)
    assert len(optimizers) == 2

    opt_g, opt_d = optimizers
    assert isinstance(opt_g, torch.optim.Optimizer)
    assert isinstance(opt_d, torch.optim.Optimizer)

    params_g = {
        id(param)
        for group in opt_g.param_groups
        for param in group["params"]
    }
    params_d = {
        id(param)
        for group in opt_d.param_groups
        for param in group["params"]
    }

    assert params_g, "generator optimizer should manage parameters"
    assert params_d, "discriminator optimizer should manage parameters"
    assert params_g.isdisjoint(params_d), "G/D optimizers must not share params"
