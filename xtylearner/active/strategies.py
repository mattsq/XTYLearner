import torch
import torch.nn as nn
from typing import Callable, Tuple


class QueryStrategy(nn.Module):
    """Base class for active learning query strategies."""

    def forward(
        self,
        model: nn.Module,
        X_unlab: torch.Tensor,
        rep_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        batch_size: int,
    ) -> torch.Tensor:
        """Return acquisition scores for ``X_unlab``."""
        raise NotImplementedError


class EntropyT(QueryStrategy):
    """Query points with high treatment entropy or log-density variance."""

    @staticmethod
    def _treatment_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        r"""Return ``p(t\mid x)`` predicted by ``model``.

        The helper tries ``predict_treatment_proba`` if available and falls
        back to common attribute names such as ``cls_t`` or ``head_T``.  When
        these heads expect concatenated outcome inputs the function appends
        zero-outcome tensors automatically.
        """
        if hasattr(model, "predict_treatment_proba"):
            try:
                return model.predict_treatment_proba(x)
            except Exception:
                pass

        def _call(head, inp):
            try:
                return head(inp)
            except Exception:
                d_y = getattr(model, "d_y", 1)
                zeros = torch.zeros(len(inp), d_y, device=inp.device)
                return head(torch.cat([inp, zeros], dim=-1))

        if hasattr(model, "cls_t"):
            logits = _call(model.cls_t, x)
        elif hasattr(model, "head_T"):
            logits = _call(model.head_T, x)
        elif hasattr(model, "C"):
            logits = _call(model.C, x)
        else:
            raise ValueError("Model does not expose treatment head")

        return logits.softmax(dim=-1)

    def forward(
        self,
        model: nn.Module,
        X_unlab: torch.Tensor,
        rep_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        batch_size: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            k = getattr(model, "k", None)
            if k is None:
                samples = []
                for _ in range(5):
                    if hasattr(model, "sample_T") and hasattr(model, "log_p_t"):
                        t = model.sample_T(X_unlab)
                        samples.append(model.log_p_t(X_unlab, t).unsqueeze(1))
                    else:
                        raise ValueError(
                            "Model must implement sample_T and log_p_t for continuous treatments"
                        )
                logp = torch.cat(samples, dim=1)
                return logp.var(dim=1)

            probs = self._treatment_proba(model, X_unlab)
            log_p = probs.clamp_min(1e-12).log()
            return -(probs * log_p).sum(dim=-1)


class DeltaCATE(QueryStrategy):
    """Query points with uncertain treatment effect."""

    def _predict_outcome(
        self, model: nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Return ``y`` predictions for covariates ``x`` and treatment ``t``.

        The method first checks for a ``predict_outcome`` implementation and
        otherwise tries a combination of ``head_Y`` and ``h`` attributes.
        """
        if hasattr(model, "predict_outcome"):
            return model.predict_outcome(x, int(t[0].item()) if t.numel() == 1 else t)
        k = getattr(model, "k", None)
        if k is not None:
            t_in = torch.nn.functional.one_hot(t.to(torch.long), k).float()
        else:
            t_in = t.unsqueeze(-1) if t.dim() == 1 else t
        if hasattr(model, "head_Y"):
            if hasattr(model, "h"):
                h = model.h(x)
                return model.head_Y(torch.cat([h, t_in], dim=-1))
            return model.head_Y(torch.cat([x, t_in], dim=-1))
        return model(x, t)

    def forward(
        self,
        model: nn.Module,
        X_unlab: torch.Tensor,
        rep_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        batch_size: int,
    ) -> torch.Tensor:
        k = getattr(model, "k", None)
        with torch.no_grad():
            preds = []
            if k is None:
                for _ in range(5):
                    if hasattr(model, "sample_T"):
                        t = model.sample_T(X_unlab)
                    else:
                        raise ValueError(
                            "Model must implement sample_T for continuous treatments"
                        )
                    preds.append(self._predict_outcome(model, X_unlab, t).unsqueeze(1))
            else:
                for t_val in range(k):
                    t = torch.full((len(X_unlab),), t_val, device=X_unlab.device)
                    preds.append(self._predict_outcome(model, X_unlab, t).unsqueeze(1))
            pred = torch.cat(preds, dim=1)
            var = pred.var(dim=1)
            return var.mean(dim=-1)


class FCCMRadius(QueryStrategy):
    """Weighted combination of entropy, variance and coverage radius."""

    def __init__(self, lambdas: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> None:
        super().__init__()
        self.lambdas = lambdas
        self._X_lab: torch.Tensor | None = None

    def update_labeled(self, X_lab: torch.Tensor) -> None:
        """Store labelled covariates for future radius computations."""
        self._X_lab = X_lab

    def _radius(self, rep_u: torch.Tensor, rep_l: torch.Tensor) -> torch.Tensor:
        """Minimum pairwise distance from unlabelled to labelled representations."""
        dists = torch.cdist(rep_u, rep_l)
        return dists.min(dim=1).values

    def forward(
        self,
        model: nn.Module,
        X_unlab: torch.Tensor,
        rep_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        batch_size: int,
    ) -> torch.Tensor:
        rep = rep_fn if rep_fn is not None else (lambda x: x)
        entropy = EntropyT()(model, X_unlab, rep_fn, batch_size)
        var = DeltaCATE()(model, X_unlab, rep_fn, batch_size)
        if self._X_lab is None:
            radius = torch.zeros_like(entropy)
        else:
            rep_u = rep(X_unlab)
            rep_l = rep(self._X_lab)
            radius = self._radius(rep_u, rep_l)
        l1, l2, l3 = self.lambdas
        return l1 * entropy + l2 * var + l3 * radius


STRATEGIES = {"entropy": EntropyT, "var": DeltaCATE, "fccm": FCCMRadius}

__all__ = ["QueryStrategy", "EntropyT", "DeltaCATE", "FCCMRadius", "STRATEGIES"]
