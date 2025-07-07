Below is a self-contained primer you can drop straight into your repo’s docs/baselines/fixmatch.md.

———

1 Motivation – “high-confidence pseudo-labels + strong consistency”

FixMatch (Sohn et al., 2020) marries two ideas that had already worked well on their own:

IdeaWhat it does
Pseudo-labellingTurn the model’s own high-confidence prediction on an unlabelled sample into a temporary label.
Consistency regularisationForce the model to give the same answer under a strong perturbation of the input.

FixMatch first gets a weakly augmented view of an unlabelled row, keeps the prediction only if its max-probability exceeds a confidence threshold \tau (typically 0.95), and then asks the model to reproduce that label on a strongly augmented view of the same row. Despite its simplicity, the method set state-of-the-art results on canonical SSL benchmarks and is still a go-to baseline in 2025 ￼.

———

2 Core algorithm (for one batch)

StepMaths (per row x)Comment
1. Weak viewp_w = \text{softmax}\bigl(f_\theta(\tilde x^{\,\text{weak}})\bigr)Light noise (or identity)
2. Pseudo‐labely^\star = \arg\max p_w,\;\;m = \mathbb 1[\max p_w \ge \tau]Keep only “confident” rows
3. Strong viewp_s = \text{softmax}\bigl(f_\theta(\tilde x^{\,\text{strong}})\bigr)Heavy corruption
4. Un-sup loss$begin:math:text$ \mathcal L_{\text{unsup}} = \frac1{\tfrac{|B_U|}{1}}p_{s} \cdots$end:math:text$
5. Total loss\mathcal L = \mathcal L_{\text{sup}} + \lambda_u\,\mathcal L_{\text{unsup}}\t\lambda_u\!\approx\!1

———

3 Augmentations for tabular data

Weak viewStrong view (choose ≥ 2)
Identity or Gaussian noise \epsilon\!\sim\!\mathcal N(0,0.01\sigma_j)• MixUp / CutMix-row across the mini-batch  • Feature dropout (mask 5–20 % of columns)  • Quantile jitter – replace a value with another sampled from the same decile  • Category swap – random replacement on categorical cols
Keep distributions similarShould visibly disturb marginal distributions

The same augmentation pipeline can be reused for Mean-Teacher or VIME, so adding it once benefits several baselines.

———

4 Reference PyTorch code (≤ 100 LoC)

```python
# fixmatch_tabular.py --------------------------------------------------
import torch, torch.nn.functional as F
from copy import deepcopy

# --- tabular augmentations -------------------------------------------
def weak_aug(x):               # light gaussian noise
    return x + 0.01 * torch.randn_like(x)

def strong_aug(x):
    # mixup  (alpha=1.0) + feature dropout
    lam = torch.distributions.Beta(1., 1.).sample((1,)).item()
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    mask = torch.rand_like(x).bernoulli_(0.15)   # 15 % drop
    noise = torch.randn_like(x) * x.std(0, keepdim=True)
    return torch.where(mask.bool(), noise, x_mix)

# --- FixMatch loss helper --------------------------------------------
def fixmatch_unsup_loss(model, x_u, τ=0.95):
    with torch.no_grad():
        p_w = F.softmax(model(weak_aug(x_u)), 1)
        max_p, y_star = p_w.max(1)
        mask = max_p.ge(τ).float()

    if mask.sum() == 0:                 # nothing confident this batch
        return torch.tensor(0., device=x_u.device)

    p_s = model(strong_aug(x_u))
    loss = F.cross_entropy(p_s, y_star, reduction='none')
    return (loss * mask).mean()

# --- training loop skeleton ------------------------------------------
def train_fixmatch(model, loader_lab, loader_unlab, epochs=200,
                   μ=7, τ=0.95, λ_u=1.0, lr=3e-4, device='cuda'):
    opt = torch.optim.AdamW(model.parameters(), lr)

    it_unlab = iter(loader_unlab)
    for epoch in range(epochs):
        for x_l, y_l in loader_lab:
            try:
                x_u = next(it_unlab)
            except StopIteration:
                it_unlab = iter(loader_unlab)
                x_u = next(it_unlab)

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u[0].to(device)     # unlabeled loader gives (x,)
            # ------------- forward
            logits_l = model(weak_aug(x_l))
            L_sup = F.cross_entropy(logits_l, y_l)

            L_unsup = fixmatch_unsup_loss(model, x_u, τ)
            loss = L_sup + λ_u * L_unsup
            # ------------- optimise
            opt.zero_grad(); loss.backward(); opt.step()
```

———

5 Drop-in wrapper for XTYLearner

```python
# xtylearner/models/fixmatch.py
from .registry import register_model
from .common_nets import build_mlp          # reuse existing encoder
from .fixmatch_tabular import train_fixmatch

@register_model("fixmatch")
class FixMatch:
    def __init__(self, τ=0.95, λ_u=1.0, μ=7):
        self.cfg = dict(τ=τ, λ_u=λ_u, μ=μ)

    def fit(self, X_lab, y_lab, X_unlab):
        import torch
        Xl = torch.tensor(X_lab, dtype=torch.float32)
        yl = torch.tensor(y_lab, dtype=torch.long)
        Xu = torch.tensor(X_unlab, dtype=torch.float32)

        net = build_mlp(Xl.size(1), yl.max()+1).to('cuda')
        lab_loader  = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xl, yl),
            batch_size=64, shuffle=True)
        unlab_loader= torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xu),
            batch_size=64*self.cfg["μ"], shuffle=True)

        train_fixmatch(net, lab_loader, unlab_loader,
                       τ=self.cfg["τ"], λ_u=self.cfg["λ_u"])
        self.net = net.eval()
        return self

    def predict_proba(self, X):
        import torch, torch.nn.functional as F
        X = torch.tensor(X, dtype=torch.float32).cuda()
        with torch.no_grad():
            return F.softmax(self.net(X), 1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
```

Add "fixmatch" to the registry and you can train via

```python
model = models.build_model("fixmatch", τ=0.9).fit(X_l, y_l, X_u)
```

———

6 Hyper-parameter heuristics

ParameterDefaultNotes
Confidence \tau0.95Lower (0.8–0.9) if labels are extremely scarce.
Un-sup weight \lambda_u1.0Up to 5 when labelled set < 100.
Ratio \mu (unlabelled : labelled in a batch)73–7 generally fine on tables.
Learning rate3 e-4 AdamWStability > SGD on small-to-medium tables.
Strong augmentation intensity10–30 % feature dropout + MixUpTune once per dataset.

———

7 Further reading & code bases
•K. Sohn et al., “FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.” NeurIPS 2020  ￼
•google-research/fixmatch – official TF code  ￼
•kekmodel/FixMatch-pytorch – concise PyTorch reference (images) for debugging  ￼

Copy the .py files into xtylearner/models/, drop this explanation into docs/baselines/fixmatch.md, and your benchmark suite now includes the last of the “Big Four” classic SSL methods.
