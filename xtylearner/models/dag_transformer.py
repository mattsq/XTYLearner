"""DAG-aware Transformer for Causal Effect Estimation.

This implementation incorporates causal DAG structure into the transformer
attention mechanism for improved treatment effect estimation.

Reference:
    Liu et al. (2024). "DAG-aware Transformer for Causal Effect Estimation."
    arXiv:2410.10044
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import make_mlp
from .registry import register_model


class DAGAwareAttention(nn.Module):
    """Multi-head attention with DAG-aware masking.

    The attention mechanism is modified to respect causal structure by masking
    attention weights according to the adjacency matrix of the causal DAG.

    Parameters
    ----------
    d_model : int
        Embedding dimension
    nhead : int
        Number of attention heads
    dropout : float
        Dropout probability
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        dag_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply DAG-aware multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, d_model)
        dag_mask : torch.Tensor, optional
            DAG adjacency mask of shape (seq_len, seq_len) where
            mask[i,j] = 1 if variable i can attend to variable j
        attn_mask : torch.Tensor, optional
            Additional attention mask (e.g., for padding)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head
        q = self.w_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # Shape: (batch, nhead, seq_len, seq_len)

        # Apply DAG-aware masking
        if dag_mask is not None:
            # Expand mask for batch and heads
            dag_mask_expanded = dag_mask.unsqueeze(0).unsqueeze(0)
            # Shape: (1, 1, seq_len, seq_len)

            # Mask out non-causal connections (set to -inf before softmax)
            scores = scores.masked_fill(dag_mask_expanded == 0, float('-inf'))

        # Apply additional attention mask if provided
        if attn_mask is not None:
            scores = scores + attn_mask

        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        # Shape: (batch, nhead, seq_len, d_k)

        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.w_o(out)

        return out


class DAGAwareTransformerLayer(nn.Module):
    """Single layer of DAG-aware transformer.

    Combines DAG-aware attention with feed-forward network and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = DAGAwareAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        dag_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention with residual
        attn_out = self.attn(self.norm1(x), dag_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x


@register_model("dag_transformer")
class DAGTransformer(nn.Module):
    """DAG-aware Transformer for treatment effect estimation.

    This model incorporates causal DAG structure into the transformer architecture,
    allowing the attention mechanism to respect known causal relationships between
    variables. Supports both known DAGs and learned DAG structures.

    Parameters
    ----------
    d_x : int
        Dimensionality of covariates X
    d_y : int
        Dimensionality of outcomes Y
    k : int
        Number of treatment categories
    d_model : int
        Transformer embedding dimension
    num_layers : int
        Number of transformer layers
    nhead : int
        Number of attention heads
    dim_feedforward : int
        Dimension of feed-forward network
    dropout : float
        Dropout probability
    learn_dag : bool
        If True, learn DAG structure from data; if False, use provided DAG
    dag_prior : torch.Tensor, optional
        Prior adjacency matrix for the DAG (shape: num_vars x num_vars)
    dag_sparsity : float
        Sparsity penalty for learned DAG (only used if learn_dag=True)
    estimator : str
        Estimation strategy: 'outcome' (G-formula), 'ipw', or 'aipw' (doubly robust)
    """

    def __init__(
        self,
        d_x: int,
        d_y: int = 1,
        k: int = 2,
        *,
        d_model: int = 128,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        learn_dag: bool = False,
        dag_prior: Optional[torch.Tensor] = None,
        dag_sparsity: float = 0.01,
        estimator: str = 'outcome',
    ):
        super().__init__()
        if d_y != 1:
            raise NotImplementedError("Currently only supports scalar outcomes (d_y=1)")
        if estimator not in ['outcome', 'ipw', 'aipw']:
            raise ValueError(f"Unknown estimator: {estimator}")

        self.d_x = d_x
        self.d_y = d_y
        self.k = k
        self.d_model = d_model
        self.num_layers = num_layers
        self.learn_dag = learn_dag
        self.dag_sparsity = dag_sparsity
        self.estimator = estimator

        # Total number of variables: d_x covariates + 1 treatment + 1 outcome
        self.num_vars = d_x + 2

        # Input embeddings for each variable type
        self.covariate_embed = nn.Linear(1, d_model)  # Embed each covariate separately
        self.treatment_embed = nn.Embedding(k + 1, d_model)  # +1 for missing indicator
        self.outcome_embed = nn.Linear(1, d_model)

        # Position embeddings
        self.pos_embed = nn.Embedding(self.num_vars, d_model)

        # DAG structure (adjacency matrix)
        if learn_dag:
            # Learnable DAG parameters (will be constrained to be DAG)
            if dag_prior is not None:
                self.dag_logits = nn.Parameter(torch.logit(dag_prior.clamp(0.01, 0.99)))
            else:
                self.dag_logits = nn.Parameter(torch.randn(self.num_vars, self.num_vars) * 0.1)
        else:
            # Use fixed DAG structure
            if dag_prior is not None:
                self.register_buffer('dag_adjacency', dag_prior)
            else:
                # Default: assume X -> T -> Y structure with no X-X edges
                dag_adjacency = self._get_default_dag()
                self.register_buffer('dag_adjacency', dag_adjacency)

        # Transformer layers
        self.layers = nn.ModuleList([
            DAGAwareTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        if estimator in ['outcome', 'aipw']:
            self.outcome_head = make_mlp([d_model, dim_feedforward // 2, 1], dropout=dropout)

        if estimator in ['ipw', 'aipw']:
            self.propensity_head = make_mlp([d_model, dim_feedforward // 2, k], dropout=dropout)

        # Combination weight for AIPW
        if estimator == 'aipw':
            self.alpha = nn.Parameter(torch.tensor(0.5))

    def _get_default_dag(self) -> torch.Tensor:
        """Create default DAG structure: X -> T -> Y with X -> Y direct effects.

        Variables are ordered as: [X_0, X_1, ..., X_{d_x-1}, T, Y]
        """
        dag = torch.zeros(self.num_vars, self.num_vars)

        # All covariates can influence treatment
        dag[:self.d_x, self.d_x] = 1.0

        # All covariates can influence outcome
        dag[:self.d_x, self.d_x + 1] = 1.0

        # Treatment influences outcome
        dag[self.d_x, self.d_x + 1] = 1.0

        # Allow self-attention (diagonal)
        dag.fill_diagonal_(1.0)

        return dag

    def get_dag_mask(self) -> torch.Tensor:
        """Get current DAG adjacency mask for attention.

        Returns
        -------
        torch.Tensor
            Binary adjacency matrix of shape (num_vars, num_vars)
        """
        if self.learn_dag:
            # Convert logits to probabilities and threshold
            dag_probs = torch.sigmoid(self.dag_logits)
            # Hard threshold for inference, soft for training
            if self.training:
                # Gumbel-softmax style continuous relaxation
                return dag_probs
            else:
                return (dag_probs > 0.5).float()
        else:
            return self.dag_adjacency

    def dag_acyclicity_loss(self) -> torch.Tensor:
        """Compute acyclicity constraint loss for learned DAG.

        Uses the trace-based constraint: trace(e^(A ∘ A)) - num_vars = 0
        for acyclic adjacency matrix A.
        """
        if not self.learn_dag:
            return torch.tensor(0.0)

        dag_probs = torch.sigmoid(self.dag_logits)
        # Remove diagonal (self-loops)
        A = dag_probs * (1 - torch.eye(self.num_vars, device=dag_probs.device))

        # Matrix exponential trace constraint
        M = torch.eye(self.num_vars, device=A.device) + A / self.num_vars
        acyclicity = torch.trace(torch.matrix_power(M, self.num_vars)) - self.num_vars

        return acyclicity

    def embed_sequence(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed variables into sequence of tokens.

        Parameters
        ----------
        x : torch.Tensor
            Covariates of shape (batch, d_x)
        t : torch.Tensor
            Treatments of shape (batch,), values in {0, ..., k-1, -1}
            where -1 indicates missing treatment
        y : torch.Tensor, optional
            Outcomes of shape (batch,) or (batch, 1)

        Returns
        -------
        torch.Tensor
            Embedded sequence of shape (batch, num_vars, d_model)
        """
        batch_size = x.size(0)
        device = x.device

        # Embed each covariate separately
        x_tokens = []
        for i in range(self.d_x):
            x_i = x[:, i:i+1]  # Shape: (batch, 1)
            x_tokens.append(self.covariate_embed(x_i))
        x_embedded = torch.stack(x_tokens, dim=1)  # (batch, d_x, d_model)

        # Embed treatment (use k as missing indicator)
        t_indices = torch.where(t == -1, torch.tensor(self.k, device=device), t)
        t_embedded = self.treatment_embed(t_indices).unsqueeze(1)  # (batch, 1, d_model)

        # Embed outcome (use zeros if not provided)
        if y is None:
            y = torch.zeros(batch_size, 1, device=device)
        elif y.dim() == 1:
            y = y.unsqueeze(1)
        y_embedded = self.outcome_embed(y).unsqueeze(1)  # (batch, 1, d_model)

        # Concatenate all tokens
        tokens = torch.cat([x_embedded, t_embedded, y_embedded], dim=1)
        # Shape: (batch, num_vars, d_model)

        # Add positional embeddings
        pos_ids = torch.arange(self.num_vars, device=device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)

        return tokens

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict outcome given covariates and treatment.

        Parameters
        ----------
        x : torch.Tensor
            Covariates of shape (batch, d_x)
        t : torch.Tensor
            Treatments of shape (batch,)

        Returns
        -------
        torch.Tensor
            Predicted outcomes of shape (batch, 1)
        """
        # Embed inputs
        tokens = self.embed_sequence(x, t, y=None)

        # Get DAG mask
        dag_mask = self.get_dag_mask()

        # Apply transformer layers
        for layer in self.layers:
            tokens = layer(tokens, dag_mask)

        # Extract outcome token representation
        y_token = tokens[:, -1, :]  # Last token is outcome

        # Predict outcome
        if self.estimator in ['outcome', 'aipw']:
            y_pred = self.outcome_head(y_token)
        else:
            # For IPW-only estimator, we still need to provide outcome predictions
            # Create a simple outcome head if needed
            if not hasattr(self, '_outcome_head_fallback'):
                self._outcome_head_fallback = make_mlp(
                    [self.d_model, 128, 1],
                    dropout=0.1
                ).to(x.device)
            y_pred = self._outcome_head_fallback(y_token)

        return y_pred

    def predict_outcome(self, x: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        """Predict outcome for all rows in ``x`` under treatment ``t``.

        Parameters
        ----------
        x : torch.Tensor
            Covariates of shape (batch, d_x)
        t : int or torch.Tensor
            Treatment value (scalar) or tensor of treatments

        Returns
        -------
        torch.Tensor
            Predicted outcomes of shape (batch,) or (batch, d_y)
        """
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)
        elif t.dim() == 0:
            t = torch.full((x.size(0),), int(t.item()), dtype=torch.long, device=x.device)

        y_pred = self.forward(x, t)
        if self.d_y == 1:
            return y_pred.squeeze(-1)
        return y_pred

    def predict_propensity(self, x: torch.Tensor) -> torch.Tensor:
        """Predict propensity scores p(T|X).

        Parameters
        ----------
        x : torch.Tensor
            Covariates of shape (batch, d_x)

        Returns
        -------
        torch.Tensor
            Propensity scores of shape (batch, k)
        """
        # Embed inputs with missing treatment
        t_missing = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        tokens = self.embed_sequence(x, t_missing, y=None)

        # Get DAG mask
        dag_mask = self.get_dag_mask()

        # Apply transformer layers
        for layer in self.layers:
            tokens = layer(tokens, dag_mask)

        # Extract treatment token representation
        t_token = tokens[:, self.d_x, :]  # Treatment token

        # Predict propensity
        if self.estimator in ['ipw', 'aipw']:
            logits = self.propensity_head(t_token)
        else:
            # For outcome-only estimator, create a propensity head if needed
            if not hasattr(self, '_propensity_head_fallback'):
                self._propensity_head_fallback = make_mlp(
                    [self.d_model, 128, self.k],
                    dropout=0.1
                ).to(x.device)
            logits = self._propensity_head_fallback(t_token)
        return F.softmax(logits, dim=-1)

    def predict_treatment_proba(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ``p(t|x)`` or ``p(t|x,y)`` depending on ``y``.

        Parameters
        ----------
        x : torch.Tensor
            Covariates of shape (batch, d_x)
        y : torch.Tensor, optional
            Outcomes of shape (batch,) or (batch, d_y)

        Returns
        -------
        torch.Tensor
            Treatment probabilities of shape (batch, k)
        """
        if y is None:
            # Return p(t|x) using propensity model
            return self.predict_propensity(x)
        else:
            # Return p(t|x,y) by embedding both x and y
            if y.dim() == 1:
                y = y.unsqueeze(1)

            t_missing = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
            tokens = self.embed_sequence(x, t_missing, y)

            # Get DAG mask
            dag_mask = self.get_dag_mask()

            # Apply transformer layers
            for layer in self.layers:
                tokens = layer(tokens, dag_mask)

            # Extract treatment token representation
            t_token = tokens[:, self.d_x, :]  # Treatment token

            # Predict p(t|x,y)
            if self.estimator in ['ipw', 'aipw']:
                logits = self.propensity_head(t_token)
            else:
                if not hasattr(self, '_propensity_head_fallback'):
                    self._propensity_head_fallback = make_mlp(
                        [self.d_model, 128, self.k],
                        dropout=0.1
                    ).to(x.device)
                logits = self._propensity_head_fallback(t_token)

            return F.softmax(logits, dim=-1)

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss.

        Parameters
        ----------
        x : torch.Tensor
            Covariates of shape (batch, d_x)
        y : torch.Tensor
            Observed outcomes of shape (batch,) or (batch, 1)
        t_obs : torch.Tensor
            Observed treatments of shape (batch,), with -1 for missing

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)

        batch_size = x.size(0)
        device = x.device

        # Mask for observed treatments
        t_observed = (t_obs != -1)

        total_loss = torch.tensor(0.0, device=device)

        # Outcome prediction loss (for observed treatments)
        if self.estimator in ['outcome', 'aipw'] and t_observed.any():
            y_pred = self.forward(x[t_observed], t_obs[t_observed])
            outcome_loss = F.mse_loss(y_pred, y[t_observed])
            total_loss = total_loss + outcome_loss

        # Propensity score loss (for observed treatments)
        if self.estimator in ['ipw', 'aipw'] and t_observed.any():
            propensity = self.predict_propensity(x[t_observed])
            propensity_loss = F.cross_entropy(
                torch.log(propensity + 1e-8),
                t_obs[t_observed]
            )
            total_loss = total_loss + propensity_loss

        # Semi-supervised loss on unlabeled data (predict treatment from X, Y)
        if (~t_observed).any():
            # Embed with observed outcomes but missing treatment
            tokens = self.embed_sequence(x[~t_observed], t_obs[~t_observed], y[~t_observed])
            dag_mask = self.get_dag_mask()

            for layer in self.layers:
                tokens = layer(tokens, dag_mask)

            # Predict treatment from representation
            t_token = tokens[:, self.d_x, :]
            if self.estimator in ['ipw', 'aipw']:
                # Use propensity head as semi-supervised prediction
                t_logits = self.propensity_head(t_token)
            else:
                # Create a temporary head for this
                if not hasattr(self, '_ssl_treatment_head'):
                    self._ssl_treatment_head = make_mlp(
                        [self.d_model, 256, self.k],
                        dropout=0.1
                    ).to(device)
                t_logits = self._ssl_treatment_head(t_token)

            # Consistency loss: predictions should be confident
            t_probs = F.softmax(t_logits, dim=-1)
            entropy_loss = -(t_probs * torch.log(t_probs + 1e-8)).sum(dim=-1).mean()
            total_loss = total_loss - 0.1 * entropy_loss  # Negative for minimization

        # DAG acyclicity constraint (if learning DAG)
        if self.learn_dag:
            acyclicity_loss = self.dag_acyclicity_loss()
            sparsity_loss = torch.sigmoid(self.dag_logits).mean()
            total_loss = total_loss + acyclicity_loss + self.dag_sparsity * sparsity_loss

        return total_loss

    @torch.no_grad()
    def predict_y(
        self,
        x_row: torch.Tensor,
        t_prompt: int,
        n_samples: int = 1,
    ) -> float:
        """Predict outcome for a single sample.

        Parameters
        ----------
        x_row : torch.Tensor
            Covariates of shape (d_x,)
        t_prompt : int
            Treatment value
        n_samples : int
            Number of samples (unused, for API compatibility)

        Returns
        -------
        float
            Predicted outcome
        """
        if x_row.dim() == 1:
            x_row = x_row.unsqueeze(0)

        t = torch.tensor([t_prompt], dtype=torch.long, device=x_row.device)
        y_pred = self.forward(x_row, t)
        return float(y_pred.squeeze().item())


__all__ = ["DAGTransformer", "DAGAwareAttention", "DAGAwareTransformerLayer"]
