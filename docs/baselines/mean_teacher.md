Below is a self-contained primer you can drop straight into your repo’s docs/baselines/mean_teacher.md.

———

1  Motivation

Mean Teacher addresses a simple question: how can we exploit un-labelled rows by forcing the model to be locally smooth, without letting confirmation bias run wild?
It does this by keeping two copies of the same network:

RoleParametersWhat they do
StudentθLearns in the usual way from labelled data.
Teacherθ′Exponential moving average (EMA) of θ; provides a slowly-moving “target” that the student must agree with on every sample.

Because θ′ changes only a little each step, it produces stable targets; forcing θ to match them yields strong regularisation and lets the un-labelled rows contribute through a consistency loss. The idea comes from Tarvainen & Valpola (NeurIPS 2017) and has become a default baseline in modern semi-supervised papers.

———

2  Theory in a nutshell

For a batch B_L of labelled rows and B_U of un-labelled rows, define

\begin{aligned}
\mathcal{L}_{\text{sup}} &= \frac1{|B_L|}\sum_{(x,y)\in B_L}\;
\mathrm{CE}\bigl(f_\theta(\tilde x),y\bigr) \\
\mathcal{L}_{\text{cons}} &= \frac1{|B_U|}\sum_{x\in B_U}\;
\lVert f_\theta(\tilde x)-f_{\theta'}(\hat x)\rVert_2^2 ,
\end{aligned}

where \tilde x,\hat x are independent noised copies of the same row.
Total loss

\mathcal{L}=\mathcal{L}_{\text{sup}}+\lambda(t)\,\mathcal{L}_{\text{cons}},

and after each optimiser step the teacher is updated

\theta' \leftarrow \alpha\,\theta' + (1-\alpha)\,\theta .

Typical values: \alpha\!\in\![0.99,0.999]. The consistency weight \lambda(t) is ramped up with a sigmoid for the first 10\u201340 epochs to prevent the student from being dominated before it has good supervised features. A reference implementation of the ramp-up appears in the original open-source code.

———

3  Algorithm (one training step)

```python
# x_lab, y_lab, x_unlab already drawn (mix them in the same batch)

# 1 Add independent noise / augmentations
x_lab_s   = add_noise(x_lab)
x_unlab_s = add_noise(x_unlab)          # student view
x_unlab_t = add_noise(x_unlab)          # teacher view

# 2 Forward passes
p_lab     = student(x_lab_s)            # logits
p_unlab_s = student(x_unlab_s)
p_unlab_t = teacher(x_unlab_t).detach() # no grads

# 3 Losses
L_sup  = CE(p_lab, y_lab)
L_cons = mse(p_unlab_s.softmax(dim=-1),
             p_unlab_t.softmax(dim=-1))
loss   = L_sup + \lambda_t * L_cons

# 4 Optimise student
loss.backward(); optimiser.step(); optimiser.zero_grad()

# 5 Update teacher weights
for \theta, \theta' in zip(student.parameters(), teacher.parameters()):
    \theta'.data.mul_(\alpha).add_(\theta.data, alpha=1-\alpha)
```

———

4  Noise recipes for tabular data

Column typeSimple noiseAlternatives
NumericAdd iid \epsilon\sim\mathcal N(0,\sigma^2) where \sigma\!=\!0.05\,\text{std(col)}.MixUp, CutMix-row, feature dropout.
CategoricalWith prob p\!=\!0.1 replace with a random category.Swap with another row in the batch (“categorical CutMix”).

Use different noise for student and teacher views; that is what gives the consistency term traction.

———

5  Drop-in PyTorch module for XTYLearner

```python
# xtylearner/models/mean_teacher.py
from copy import deepcopy
import torch, torch.nn as nn
from .registry import register_model

@register_model("mean_teacher")
class MeanTeacher(nn.Module):
    def __init__(self, base_net_fn, num_classes,
                 ema_decay=0.99, ramp_up=40, cons_max=1.0):
        super().__init__()
        self.student  = base_net_fn(num_classes)
        self.teacher  = deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.ema_decay  = ema_decay
        self.ramp_up    = ramp_up
        self.cons_max   = cons_max
        self.step       = 0          # global step counter

    def _consistency_weight(self, epoch):
        t = min(epoch / self.ramp_up, 1.0)
        return self.cons_max * torch.exp(-5 * (1 - t) ** 2)

    def forward(self, x, teacher=False):
        net = self.teacher if teacher else self.student
        return net(x)

    def update_teacher(self):
        \alpha = self.ema_decay
        for \theta_s, \theta_t in zip(self.student.parameters(),
                          self.teacher.parameters()):
            \theta_t.data.mul_(\alpha).add_(\theta_s.data, alpha=1-\alpha)
```

Training loop (simplified):

```python
for epoch in range(epochs):
    for (x_lab, y_lab), (x_unlab,) in loader:
        x_lab_s   = noise(x_lab);   x_unlab_s = noise(x_unlab)
        x_unlab_t = noise(x_unlab)

        logits_lab = model.student(x_lab_s)
        logits_s   = model.student(x_unlab_s)
        with torch.no_grad():
            logits_t = model.teacher(x_unlab_t)

        L_sup  = ce_loss(logits_lab, y_lab)
        L_cons = mse_loss(logits_s.softmax(-1),
                          logits_t.softmax(-1))
        \lambda  = model._consistency_weight(epoch)
        loss   = L_sup + \lambda * L_cons

        loss.backward(); opt.step(); opt.zero_grad()
        model.update_teacher()
```

During inference use the teacher for slightly better calibration.

———

6  Hyperparameter tips

KnobGood starting point
ema_decay \alpha0.99 (small labels) \u2192 0.999 (large labels)
Consistency weight \lambda_max1.0 for balanced classes; up to 50 on very small label sets
Ramp-up epochs10 \u2013 40
Batch mix1 : 3 labelled : unlabelled often works well
OptimiserAdamW, lr = 3 e-4, weight-decay = 1 e-2

Watch out for confirmation bias if you set \lambda too large or ramp up too fast; the student can collapse to the teacher’s early mistakes.

———

7  Further reading
•Antti Tarvainen & Harri Valpola, “Mean teachers are better role models”, NeurIPS 2017
•CuriousAI reference implementation (PyTorch)
•Practical hyper-parameter guide and code snippets
•Ramp-up utility functions (open-source)
