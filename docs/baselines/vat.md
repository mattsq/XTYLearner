Below is a self-contained primer you can drop straight into your repo’s docs/baselines/vat.md.

———

1  Motivation—“smoothness, but only where it counts”

Adversarial‑training tells a model to be robust against perturbations that maximally change its prediction.
Virtual Adversarial Training (VAT) takes that idea, but drops the need for labels: it finds, for each input x, the tiny direction r_{adv} that would most distort the model’s output distribution and then forces the network to agree with itself even there.
•100 % of the batch (labelled + unlabelled) therefore supplies a “consistency” signal.
•One extra forward–backward pass gives a very strong regulariser that works on images, text and plain tables (because it does not rely on domain‑specific augmentations).
The method was introduced by Miyato et al. (ICLR 2018)  and is still a staple baseline in semi-supervised benchmarks and tabular SSL libraries.

———

2  Theory in two steps

symbolmeaning
p_\theta(y\mid x)model posterior (softmax)
\text{KL}(p\Vert q)Kullback-Leibler divergence
\epsilonradius of adversarial ball (e.g. 2–10 % of feature std)
\xismall constant for finite-difference (usually 10^{-6})

1.Find the adversarial direction
r_{\text{adv}} = \arg\max_{\lVert r\rVert_2\le\epsilon}
\text{KL}\!\bigl(p_\theta(y\mid x)\;\Vert\;p_\theta(y\mid x+r)\bigr)
Because the inner problem is intractable, use one power-iteration:

g  = ∇_r  KL(pθ(x) ‖ pθ(x+r)) |_{r = ξ·d},
d′ = g / ‖g‖₂,   r_adv = ε·d′

with d an iid unit Gaussian vector.

2.Consistency loss
\mathcal L_{\text{vat}}
= \text{KL}\!\bigl(p_\theta(y\mid x)\;\Vert\;p_\theta(y\mid x+r_{\text{adv}})\bigr)
Add this to ordinary cross-entropy on the labelled rows (weight \lambda(t) is usually ramped from 0 → 1 over the first 10–40 epochs, exactly like Mean Teacher).

———

3  Reference PyTorch implementation (≤60 LoC)

```python
# vat.py
import torch, torch.nn.functional as F

def _l2_normalise(d):  # helper
    return d / (torch.norm(d.view(d.size(0), -1), dim=1, keepdim=True) + 1e-8)

def vat_loss(model, x, xi=1e-6, eps=2.5, n_power=1):
    with torch.no_grad():
        pred = F.softmax(model(x), 1)

    # 1) generate adversarial direction
    d = torch.randn_like(x)
    for _ in range(n_power):
        d = xi * _l2_normalise(d)
        d.requires_grad_()
        pred_hat = model(x + d)
        adv_dist = F.kl_div(
            F.log_softmax(pred_hat, 1), pred, reduction="batchmean")
        grad = torch.autograd.grad(adv_dist, d)[0]
        d = grad.detach()            # power-iter step

    r_adv = eps * _l2_normalise(d)

    # 2) actual VAT loss (no back-prop through pred)
    pred_hat = model(x + r_adv)
    loss = F.kl_div(
        F.log_softmax(pred_hat, 1), pred, reduction="batchmean")
    return loss
```

Training step (inside your loop):

```python
# x_lab, y_lab, x_unlab  already sampled
logits_lab = model(x_lab)
L_sup = F.cross_entropy(logits_lab, y_lab)

# VAT on both lab+unlab
x_all = torch.cat([x_lab, x_unlab], 0)
L_vat = vat_loss(model, x_all, eps=2.5)

λ = ramp_up(epoch, max_val=1.0)        # e.g. sigmoid ramp
loss = L_sup + λ * L_vat
loss.backward(); optimiser.step(); optimiser.zero_grad()
```

———

4  Drop-in wrapper for XTYLearner

```python
# xtylearner/models/vat.py
from .registry import register_model
from .utils import ramp_up_sigmoid               # helper you likely have
from .common_nets import build_mlp               # reuse existing encoder

@register_model("vat")
class VAT_Model:
    def __init__(self, eps=2.5, xi=1e-6, n_power=1, λ_max=1.0, ramp=30):
        self.cfg = dict(eps=eps, xi=xi, n_power=n_power,
                        λ_max=λ_max, ramp=ramp)

    def fit(self, X_lab, y_lab, X_unlab, epochs=200, bs=256):
        import torch, torch.nn as nn
        Xl = torch.tensor(X_lab, dtype=torch.float32)
        yl = torch.tensor(y_lab, dtype=torch.long)
        Xu = torch.tensor(X_unlab, dtype=torch.float32)

        self.net = build_mlp(Xl.size(1), yl.max()+1)
        opt = torch.optim.AdamW(self.net.parameters(), 3e-4)

        for epoch in range(epochs):
            # --- sample mixed batch
            idx_l = torch.randint(0, len(Xl), (bs,))
            idx_u = torch.randint(0, len(Xu), (bs*3,))
            x_l, y_l = Xl[idx_l], yl[idx_l]
            x_u = Xu[idx_u]
            # --- forward
            logits = self.net(x_l)
            L_sup = nn.CrossEntropyLoss()(logits, y_l)
            L_vat = vat_loss(self.net, torch.cat([x_l, x_u]),
                             xi=self.cfg["xi"],
                             eps=self.cfg["eps"],
                             n_power=self.cfg["n_power"])
            λ = ramp_up_sigmoid(epoch, self.cfg["ramp"], self.cfg["λ_max"])
            loss = L_sup + λ * L_vat
            # --- optimise
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        import torch, torch.nn.functional as F
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return F.softmax(self.net(X), 1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
```

Registering "vat" gives you:

```python
model = models.build_model("vat", eps=3.0).fit(X_l, y_l, X_u)
```

———

5  Hyper-parameter heuristics

knobusual rangecomment
ε (radius)1 – 5 × feature-stdlarger when data are well-scaled
ξ (finite diff)1e-6 … 1e-5rarely needs tuning
n_power1≥2 helps on images, negligible on tables
λ_max0.5 – 3weigh vs label scarcity
batch mix≥3 × more unlabelled rows than labelledgives stable KL estimates

———

6  Further reading
•Miyato et al., “Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning”, ICLR 2018
•lyakaap/VAT-pytorch – concise reference code
•Colab tutorial (NNabla) with visualisations of r_{adv} trajectories
