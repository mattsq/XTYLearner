import logging
from typing import Callable, Tuple

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def _predict_outcome(
    model: nn.Module, x: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Return ``y`` predictions for covariates ``x`` and treatment ``t``.

    The helper first checks for a ``predict_outcome`` implementation and
    otherwise tries a combination of ``head_Y`` and ``h`` attributes.  When no
    specialised head is found the model is called directly.  This mirrors the
    logic previously embedded in :class:`DeltaCATE` and is shared with
    :class:`CATEUncertainty`.
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
        """Return ``p(t\mid x)`` predicted by ``model``.

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
                        raise ValueError("Model must implement sample_T for continuous treatments")
                    preds.append(_predict_outcome(model, X_unlab, t).unsqueeze(1))
            else:
                for t_val in range(k):
                    t = torch.full((len(X_unlab),), t_val, device=X_unlab.device)
                    preds.append(_predict_outcome(model, X_unlab, t).unsqueeze(1))
            pred = torch.cat(preds, dim=1)
            var = pred.var(dim=1)
            return var.mean(dim=-1)


class CATEUncertainty(QueryStrategy):
    r"""Rank samples by uncertainty in their conditional treatment effect.

    Motivation
    ----------
    1. ``τ(x) = E[Y\mid X=x, T=1] - E[Y\mid X=x, T=0]`` directly determines
       which treatment a policy will recommend.  Reducing uncertainty in
       ``τ(x)`` therefore targets the decision metric instead of factual outcome
       error.
    2. Uncertain ``τ(x)`` implies high potential regret for the greedy policy
       ``t*(x) = argmax_t E[Y\mid X=x, T=t]``.  Querying those points focuses
       the labelling budget on actionable ambiguities.
    3. Because only factual triples ``(x, t_{obs}, y_{obs})`` are observable the
       strategy relies on modelled potential outcomes and Monte Carlo sampling
       to approximate the epistemic variance of ``τ(x)``.
    """

    def __init__(
        self,
        mc_samples: int = 10,
        *,
        fallback: QueryStrategy | None = None,
    ) -> None:
        super().__init__()
        self.mc_samples = mc_samples
        self._fallback = fallback

    def _ensure_tensor(
        self, value: torch.Tensor | float | list | tuple, device: torch.device
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return torch.as_tensor(value, device=device, dtype=torch.float32)

    def _reduce_tau(self, tau: torch.Tensor) -> torch.Tensor:
        if tau.dim() == 1:
            return tau
        dims = tuple(range(1, tau.dim()))
        return tau.mean(dim=dims)

    def _predict_cate(
        self, model: nn.Module, X_unlab: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if hasattr(model, "predict_cate"):
            try:
                result = model.predict_cate(X_unlab, return_uncertainty=True)
            except TypeError:
                # Older implementations may not accept the keyword flag.
                try:
                    tau_mean = model.predict_cate(X_unlab)
                except Exception:
                    pass
                else:
                    return tau_mean, None
            except NotImplementedError:
                result = None
            except Exception:
                logger.debug("predict_cate failed; falling back to MC estimate", exc_info=True)
                result = None
            else:
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                return result, None

        return self._mc_cate(model, X_unlab)

    def _mc_cate(
        self, model: nn.Module, X_unlab: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        k = getattr(model, "k", None)
        if k is None:
            logger.warning(
                "CATE uncertainty strategy received a model without discrete "
                "treatment heads; defaulting to fallback strategy."
            )
            return None, None
        if k != 2:
            logger.warning(
                "CATE uncertainty strategy currently supports binary treatments; got k=%s", k
            )
            return None, None

        mc = max(1, self.mc_samples)
        device = X_unlab.device
        t0 = torch.zeros(len(X_unlab), device=device, dtype=torch.long)
        t1 = torch.ones(len(X_unlab), device=device, dtype=torch.long)

        model_state = model.training
        model.train(True)
        tau_samples = []
        try:
            with torch.no_grad():
                for _ in range(mc):
                    y1 = _predict_outcome(model, X_unlab, t1)
                    y0 = _predict_outcome(model, X_unlab, t0)
                    tau_samples.append(
                        self._reduce_tau((y1 - y0).reshape(len(X_unlab), -1))
                    )
        except Exception:
            logger.warning(
                "Unable to estimate potential outcomes for CATE uncertainty; "
                "using fallback strategy.",
                exc_info=True,
            )
            model.train(model_state)
            return None, None
        model.train(model_state)

        tau_stack = torch.stack(tau_samples, dim=0)
        tau_mean = tau_stack.mean(dim=0)
        if mc == 1:
            tau_var = torch.zeros_like(tau_mean)
        else:
            tau_var = tau_stack.var(dim=0, unbiased=False)
        return tau_mean, tau_var

    def _fallback_strategy(self) -> QueryStrategy:
        if self._fallback is None:
            self._fallback = EntropyT()
        return self._fallback

    def forward(
        self,
        model: nn.Module,
        X_unlab: torch.Tensor,
        rep_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        batch_size: int,
    ) -> torch.Tensor:
        tau_mean, tau_unc = self._predict_cate(model, X_unlab)

        if tau_mean is None or tau_unc is None:
            fallback = self._fallback_strategy()
            logger.warning(
                "Falling back to %s for CATE uncertainty acquisition.",
                fallback.__class__.__name__,
            )
            try:
                return fallback(model, X_unlab, rep_fn, batch_size)
            except Exception:
                logger.warning(
                    "Fallback strategy %s failed; returning zero scores.",
                    fallback.__class__.__name__,
                    exc_info=True,
                )
                return torch.zeros(len(X_unlab), device=X_unlab.device)

        device = X_unlab.device
        tau_mean = self._ensure_tensor(tau_mean, device)
        tau_unc = self._ensure_tensor(tau_unc, device)

        tau_mean = self._reduce_tau(tau_mean)
        if tau_unc.dim() > 1:
            dims = tuple(range(1, tau_unc.dim()))
            tau_unc = tau_unc.mean(dim=dims)

        if torch.allclose(tau_unc, torch.zeros_like(tau_unc)):
            tau_unc = tau_mean.abs()

        return tau_unc


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


STRATEGIES = {
    "entropy": EntropyT,
    "var": DeltaCATE,
    "fccm": FCCMRadius,
    "cate_uncertainty": CATEUncertainty,
}

__all__ = [
    "QueryStrategy",
    "EntropyT",
    "DeltaCATE",
    "CATEUncertainty",
    "FCCMRadius",
    "STRATEGIES",
]
