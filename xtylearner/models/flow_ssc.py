# flow_ssc.py
# Dinh et al. “Density Estimation using Real NVP”, ICLR 2017 
# arxiv.org
# Papamakarios et al. “Masked Autoregressive Flow”, NeurIPS 2017 
# arxiv.org
# Liao et al. “SSCFlow: Semi-Supervised Conditional Normalizing Flow”, KBS 2023 (proposes the labelled + unlabelled ELBO we used) 
# sciencedirect.com
# Ardizzone et al. “Conditional Invertible Neural Networks”, CVPR 2019 
# arxiv.org

import torch, torch.nn as nn
from nflows.distributions import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms import CompositeTransform, ReversePermutation, AffineCouplingTransform
from nflows.nn.nets import ResidualNet

from .registry import register_model


def make_conditional_flow(dim_xy: int, context_dim: int,
                          n_layers: int = 6, hidden: int = 128) -> Flow:
    """Real-NVP with context (one-hot T) injected in every coupling net."""
    transforms = []
    mask = torch.arange(dim_xy) % 2
    for _ in range(n_layers):
        transforms += [
            AffineCouplingTransform(
                mask,
                transform_net_create_fn=lambda c_in, c_out: ResidualNet(
                    c_in, c_out,
                    hidden_features=hidden,
                    context_features=context_dim)),
            ReversePermutation(dim_xy)
        ]
        mask = 1 - mask  # flip mask each block
    return Flow(CompositeTransform(transforms), StandardNormal([dim_xy]))


@register_model("flow_ssc")
class MixtureOfFlows(nn.Module):
    """
    * one conditional flow  p_theta(x,y | t)  (context = one-hot(t))
    * one classifier        p_psi(t | x)
    """
    def __init__(self, dim_x, dim_y, n_treat):
        super().__init__()
        self.n_treat = n_treat
        self.flow = make_conditional_flow(dim_x + dim_y, n_treat)
        self.clf  = nn.Sequential(
            nn.Linear(dim_x, 128), nn.ReLU(),
            nn.Linear(128, 128),   nn.ReLU(),
            nn.Linear(128, n_treat))
        
    # ---------- log-likelihood for a minibatch --------------------------
    def loss(self, X, Y, T_obs):
        """T_obs = int in [0,K-1] for labelled rows, -1 for missing."""
        B = X.size(0)
        t_lab_mask = T_obs >= 0
        t_ulb_mask = ~t_lab_mask
        
        # ---- labelled part --------------------------------------------
        loss_lab = torch.tensor(0., device=X.device)
        if t_lab_mask.any():
            t_lab     = T_obs[t_lab_mask]
            ctx_lab   = torch.nn.functional.one_hot(t_lab, self.n_treat).float()
            xy_lab    = torch.cat([X[t_lab_mask], Y[t_lab_mask]], dim=-1)
            
            ll_flow   = self.flow.log_prob(xy_lab, context=ctx_lab)
            ce_clf    = nn.functional.cross_entropy(
                           self.clf(X[t_lab_mask]), t_lab)
            loss_lab  = -(ll_flow.mean() - ce_clf)    # maximise ll_flow
            
        # ---- un-labelled part -----------------------------------------
        loss_ulb = torch.tensor(0., device=X.device)
        if t_ulb_mask.any():
            X_u, Y_u = X[t_ulb_mask], Y[t_ulb_mask]
            logits   = self.clf(X_u)                  # (B_u,K)
            log_p_t  = logits.log_softmax(-1)         # log p_psi(T|X)
            
            # flow likelihood under each treatment (batch, K)
            xy_u = torch.cat([X_u, Y_u], dim=-1).unsqueeze(1)     # (B_u,1,D)
            ctx  = torch.eye(self.n_treat, device=X.device)\
                     .unsqueeze(0).repeat(X_u.size(0),1,1)        # (B_u,K,K)
            ll   = self.flow.log_prob(
                      xy_u.expand(-1, self.n_treat, -1)
                          .reshape(-1, xy_u.size(-1)),
                      context=ctx.reshape(-1, self.n_treat)
                   ).view(X_u.size(0), self.n_treat)              # (B_u,K)
            
            # log p(x,y) = logsumexp_t [ log p(t|x) + log p(x,y|t) ]
            lse = torch.logsumexp(log_p_t + ll, dim=-1)
            loss_ulb = -lse.mean()              # maximise log-evidence
            
        return (loss_lab + loss_ulb)
