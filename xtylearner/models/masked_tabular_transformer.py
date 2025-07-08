"""Masked-token Transformer for joint tabular modelling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class NumEmbed(nn.Module):
    """Project a scalar feature into a token embedding."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lin = nn.Linear(1, d_model)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.lin(value.unsqueeze(-1))


@register_model("masked_tabular_transformer")
class MaskedTabularTransformer(nn.Module):
    """Transformer encoder trained with a masked-token objective."""

    CLS, TOK_T, TOK_Y = 0, 1, 2

    def __init__(
        self,
        d_x: int,
        *,
        y_bins: int = 32,
        d_model: int = 128,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 256,
        p_mask: float = 0.15,
    ) -> None:
        super().__init__()
        self.d_x = d_x
        self.y_bins = y_bins
        self.d_model = d_model
        self.p_mask = p_mask
        self.seq_len = d_x + 5

        self.num_emb = nn.ModuleList([NumEmbed(d_model) for _ in range(d_x)])
        self.tok_T = nn.Embedding(3, d_model)
        self.tok_Y = nn.Embedding(y_bins + 1, d_model)
        self.tok_special = nn.Embedding(3, d_model)
        self.pos_emb = nn.Embedding(d_x + 6, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head_T = nn.Linear(d_model, 3)
        self.head_Y = nn.Linear(d_model, y_bins)

        self.register_buffer("y_min", torch.tensor(0.0))
        self.register_buffer("y_max", torch.tensor(1.0))

    # ------------------------------------------------------------------
    def set_y_range(self, y_min: float, y_max: float) -> None:
        """Set min/max outcome values for discretisation."""

        self.y_min.fill_(float(y_min))
        self.y_max.fill_(float(y_max))

    # ------------------------------------------------------------------
    def _discretise_y(self, y: torch.Tensor) -> torch.Tensor:
        y_norm = (y - self.y_min) / (self.y_max - self.y_min + 1e-8)
        return (y_norm * (self.y_bins - 1)).round().long().clamp(0, self.y_bins - 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return outcome logits ``p(y|x,t)`` from the encoder."""

        b = x.size(0)
        device = x.device
        seqs = []
        for i in range(b):
            row = self.row_to_tokens(
                x[i], torch.tensor(float("nan"), device=device), t[i]
            )
            seqs.append(row)
        tok_matrix = torch.stack(seqs)
        idx = torch.arange(self.seq_len, device=device)
        out = self.encoder((tok_matrix + self.pos_emb(idx)).transpose(0, 1)).transpose(0, 1)
        logits = self.head_Y(out[:, self.d_x + 4])
        return logits

    # ------------------------------------------------------------------
    def row_to_tokens(
        self, x_row: torch.Tensor, y_disc: torch.Tensor, t_val: torch.Tensor
    ) -> torch.Tensor:
        device = x_row.device
        tokens = [self.tok_special(torch.tensor([self.CLS], device=device))]
        for i, val in enumerate(x_row):
            tokens.append(self.num_emb[i](val.unsqueeze(0)))
        tokens.append(self.tok_special(torch.tensor([self.TOK_T], device=device)))
        t_tok = 2 if int(t_val) == -1 else int(t_val)
        tokens.append(self.tok_T(torch.tensor([t_tok], device=device)))
        tokens.append(self.tok_special(torch.tensor([self.TOK_Y], device=device)))
        if torch.isnan(y_disc):
            y_tok = self.y_bins
        else:
            y_tok = int(y_disc)
        tokens.append(self.tok_Y(torch.tensor([y_tok], device=device)))
        return torch.cat(tokens, 0)

    # ------------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, y: torch.Tensor, t_obs: torch.Tensor
    ) -> torch.Tensor:
        b = x.size(0)
        device = x.device
        y_disc = self._discretise_y(y.view(-1))

        tok_matrix = torch.zeros(b, self.seq_len, self.d_model, device=device)
        labels_T = torch.full((b,), -100, dtype=torch.long, device=device)
        labels_Y = torch.full((b,), -100, dtype=torch.long, device=device)

        for i in range(b):
            row = self.row_to_tokens(x[i], y_disc[i], t_obs[i])
            tok_matrix[i] = row
            labels_T[i] = t_obs[i] if t_obs[i] != -1 else -100
            labels_Y[i] = y_disc[i]

        mask_mask = torch.rand(b, self.seq_len, device=device) < self.p_mask
        mask_mask[:, 0] = False
        mask_mask[:, self.d_x + 1] = False
        mask_mask[:, self.d_x + 3] = False
        mask_mask[t_obs == -1, self.d_x + 2] = True

        mask_vec = self.tok_T(torch.tensor([2], device=device))
        tok_matrix[mask_mask] = mask_vec

        idx = torch.arange(self.seq_len, device=device)
        tok_matrix = tok_matrix + self.pos_emb(idx)
        out = self.encoder(tok_matrix.transpose(0, 1)).transpose(0, 1)

        logits_T = self.head_T(out[:, self.d_x + 2])
        logits_Y = self.head_Y(out[:, self.d_x + 4])

        loss_T = F.cross_entropy(logits_T, labels_T, ignore_index=-100)
        loss_Y = F.cross_entropy(logits_Y, labels_Y)
        return loss_T + loss_Y

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_y(
        self, x_row: torch.Tensor, t_prompt: int, n_samples: int = 20
    ) -> float:
        device = x_row.device
        idx = torch.arange(self.seq_len, device=device)
        y_probs = torch.zeros(self.y_bins, device=device)
        for _ in range(n_samples):
            seq = self.row_to_tokens(
                x_row, torch.tensor(float("nan"), device=device), t_prompt
            )
            seq[self.d_x + 4] = self.tok_Y(torch.tensor([self.y_bins], device=device))
            seq = seq + self.pos_emb(idx)
            out = self.encoder(seq.unsqueeze(1)).squeeze(1)
            logits = self.head_Y(out[self.d_x + 4])
            y_probs += F.softmax(logits, dim=0)
        y_probs /= n_samples
        pred_bin = y_probs.argmax().item()
        y_val = self.y_min + pred_bin / (self.y_bins - 1) * (self.y_max - self.y_min)
        return float(y_val)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_treatment_proba(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``p(t|x,y)`` by running the input through the encoder."""

        b = x.size(0)
        device = x.device
        y_disc = self._discretise_y(y.view(-1))
        seqs = []
        for i in range(b):
            row = self.row_to_tokens(x[i], y_disc[i], torch.tensor(-1, device=device))
            seqs.append(row)
        tok_matrix = torch.stack(seqs)
        idx = torch.arange(self.seq_len, device=device)
        out = self.encoder((tok_matrix + self.pos_emb(idx)).transpose(0, 1)).transpose(0, 1)
        logits = self.head_T(out[:, self.d_x + 2])
        return logits.softmax(dim=-1)


__all__ = ["MaskedTabularTransformer"]
