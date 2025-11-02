import torch

from xtylearner.active import CATEUncertainty, EntropyT


class _DummyCATEModel(torch.nn.Module):
    def __init__(self, tau: torch.Tensor, unc: torch.Tensor) -> None:
        super().__init__()
        self._tau = tau
        self._unc = unc

    def predict_cate(self, X: torch.Tensor, return_uncertainty: bool = False):
        if return_uncertainty:
            return self._tau.to(X.device), self._unc.to(X.device)
        return self._tau.to(X.device)


def test_cate_uncertainty_scores_match_model_uncertainty():
    X = torch.zeros(3, 2)
    tau = torch.tensor([0.2, -0.1, 0.4])
    unc = torch.tensor([0.05, 0.8, 0.1])
    model = _DummyCATEModel(tau, unc)

    strategy = CATEUncertainty()
    scores = strategy(model, X, None, batch_size=2)

    assert torch.allclose(scores, unc)
    assert torch.argmax(scores).item() == 1


def test_cate_uncertainty_falls_back_to_abs_tau_when_uncertainty_zero():
    X = torch.zeros(3, 2)
    tau = torch.tensor([0.2, -5.0, 0.1])
    unc = torch.zeros_like(tau)
    model = _DummyCATEModel(tau, unc)

    strategy = CATEUncertainty()
    scores = strategy(model, X, None, batch_size=2)

    expected = tau.abs()
    assert torch.allclose(scores, expected)


def test_cate_uncertainty_returns_zero_scores_when_no_fallback_possible():
    class NoCATEModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k = None

        def predict_treatment_proba(self, X: torch.Tensor) -> torch.Tensor:
            probs = torch.tensor(
                [[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32
            )
            return probs.to(X.device)

    X = torch.zeros(3, 2)
    model = NoCATEModel()
    entropy = EntropyT()

    strategy = CATEUncertainty(fallback=entropy)
    scores = strategy(model, X, None, batch_size=2)

    assert torch.all(scores == 0)


def test_cate_uncertainty_falls_back_to_entropy_when_available():
    class DiscreteNoCATEModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k = 2

        def predict_outcome(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        def predict_treatment_proba(self, X: torch.Tensor) -> torch.Tensor:
            probs = torch.tensor(
                [[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float32
            )
            return probs.to(X.device)

    X = torch.zeros(3, 2)
    model = DiscreteNoCATEModel()
    entropy = EntropyT()

    strategy = CATEUncertainty(fallback=entropy)
    scores = strategy(model, X, None, batch_size=2)

    expected = entropy(model, X, None, batch_size=2)
    assert torch.allclose(scores, expected)


def test_cate_uncertainty_uses_monte_carlo_when_no_direct_interface():
    class RandomOutcomeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k = 2
            self._generator = torch.Generator()
            self._generator.manual_seed(0)

        def predict_outcome(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            noise = torch.randn(len(X), generator=self._generator, device=X.device)
            return t.float() + noise

    X = torch.zeros(4, 3)
    manual_model = RandomOutcomeModel()
    tau_samples = []
    for _ in range(3):
        ones = torch.ones(len(X), dtype=torch.long, device=X.device)
        zeros = torch.zeros(len(X), dtype=torch.long, device=X.device)
        y1 = manual_model.predict_outcome(X, ones)
        y0 = manual_model.predict_outcome(X, zeros)
        tau_samples.append((y1 - y0).reshape(len(X), -1).mean(dim=1))
    tau_stack = torch.stack(tau_samples, dim=0)
    manual_var = tau_stack.var(dim=0, unbiased=False)

    model = RandomOutcomeModel()
    strategy = CATEUncertainty(mc_samples=3)
    scores = strategy(model, X, None, batch_size=2)

    assert torch.all(scores > 0)
    assert torch.allclose(scores, manual_var, atol=1e-6)
