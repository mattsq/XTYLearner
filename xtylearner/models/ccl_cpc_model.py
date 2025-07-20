import torch
from torch import nn
import torch.nn.functional as F
from .registry import register_model
@register_model("ccl_cpc")
class CCL_CPCModel(nn.Module):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        k: int,
        hidden: int = 128,
        lambda_cpc: float = 1.0,
        lambda_y: float = 1.0,
        lambda_t: float = 0.1,
        temperature: float = 0.07,
        k_future: int = 3,
        seq_len: int = 1,
    ) -> None:
        super().__init__()
        self.k = k
        self.d_y = d_y
        self.tau = temperature
        self.k_future = k_future
        self.w = {"cpc": lambda_cpc, "y": lambda_y, "t": lambda_t}
        self.enc = nn.Sequential(nn.Linear(d_x, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.rnn = None if seq_len == 1 else nn.GRU(hidden, hidden, batch_first=True)
        self.out_head = nn.Sequential(nn.Linear(hidden + k, hidden), nn.ReLU(), nn.Linear(hidden, 2 * d_y))
        self.t_head = nn.Sequential(nn.Linear(hidden + d_y, hidden), nn.ReLU(), nn.Linear(hidden, k))
    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            z = self.enc(x).unsqueeze(1)
        else:
            b, l, d = x.shape
            z = self.enc(x.view(-1, d)).view(b, l, -1)
        if self.rnn:
            z, _ = self.rnn(z)
        return z, z[:, -1]
    def _info_nce(self, z: torch.Tensor) -> torch.Tensor:
        a, p, h = z[:, :-self.k_future], z[:, self.k_future:], z.size(-1)
        sim = a.reshape(-1, h) @ p.reshape(-1, h).T / self.tau
        return F.cross_entropy(sim, torch.arange(sim.size(0), device=z.device))
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, z = self._encode(x)
        oh = F.one_hot(t.to(torch.long), self.k).float()
        mu, _ = self.out_head(torch.cat([z, oh], -1)).chunk(2, -1)
        return mu
    def loss(self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
        seq, z = self._encode(x)
        lcpc = self._info_nce(seq) if seq.size(1) > self.k_future else 0.0
        t1h = F.one_hot(t_obs.clamp_min(0), self.k).float()
        mu, log_s = self.out_head(torch.cat([z, t1h], -1)).chunk(2, -1)
        lab = t_obs >= 0
        ly = (0.5 * ((y - mu) / log_s.exp()).pow(2) + log_s).sum(-1)
        ly = (ly * lab).mean()
        lt = torch.tensor(0.0, device=x.device)
        if lab.any():
            logits = self.t_head(torch.cat([z[lab], y[lab]], -1))
            lt = F.cross_entropy(logits, t_obs[lab])
        return self.w["cpc"] * lcpc + self.w["y"] * ly + self.w["t"] * lt
    @torch.no_grad()
    def predict_outcome(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(x, t)
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, z = self._encode(x)
        return self.t_head(torch.cat([z, y], -1)).softmax(-1)
    @torch.no_grad()
    def counterfactual(self, x: torch.Tensor) -> torch.Tensor:
        _, z = self._encode(x)
        eye = torch.eye(self.k, device=x.device)
        z = z.unsqueeze(1).expand(-1, self.k, -1)
        mu, _ = self.out_head(torch.cat([z, eye.expand(z.size(0), -1, -1)], -1)).chunk(2, -1)
        return mu
__all__ = ["CCL_CPCModel"]
