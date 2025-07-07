Below is a self-contained primer you can drop straight into your repo’s docs/baselines/label_propagation.md.

———

1  Why bother with Label Propagation?
•Manifold assumption. If points are close in input space, they probably share a label. A graph turns that intuition into a global smoothness constraint.
•“Free” use of unlabelled rows. Unlike self-supervised pre-text methods that still require a supervised fine-tune, LP directly predicts labels for every unlabelled sample—often beating small neural nets when labels ≪ 1 000.
•Computationally cheap. With a k-NN graph LP is O(k n) per iteration and converges in ≼ 20 iterations for typical tabular data; you can fit it in seconds with scikit-learn.
•Interpretable upper bound. Because it ignores model capacity, LP tells reviewers “how far you can get just by exploiting geometry”.

The algorithm traces back to Zhu & Ghahramani (2002) and its probabilistic reinterpretation as a Gaussian Random Field (Zhu, Ghahramani & Lafferty 2003)  👤 👤.

———

2  Theory in two equations
1.Graph construction

W_{ij}=k_\sigma(x_i,x_j)=\exp\!\Bigl(-\tfrac{\lVert x_i-x_j\rVert^{2}}{2\sigma^{2}}\Bigr)
or a binary k-NN adjacency.
Create the (row-)normalised similarity matrix

S=D^{-1} W,\qquad D_{ii}=\sum_j W_{ij}.
2.Harmonic energy minimisation

Keep the label matrix Y\in\{0,1\}^{n\times C} fixed on labelled nodes \mathcal L and minimise

\mathcal E(F)=\tfrac12\sum_{i,j} W_{ij}\lVert F_i-F_j\rVert^{2}\quad\text{s.t. }F_i=Y_i\,(i\!\in\!\mathcal L).

The stationary solution satisfies the harmonic condition F_i=\sum_j S_{ij}F_j for unlabelled nodes. Rearranging yields the closed form

F_{\mathcal U}=(I-S_{\mathcal UU})^{-1}S_{\mathcal UL}Y_{\mathcal L}.

In practice scikit-learn iterates

F^{(t+1)}=\alpha S F^{(t)}+(1-\alpha)Y,

re-clamping the labelled rows each step; \alpha\!=\!1 gives Zhu’s original “Label Propagation”, 0<\alpha<1 gives “Label Spreading” (Zhou et al. 2004) that adds Laplacian smoothing.

———

3  Implementation guide

3.1 Quick-start with scikit-learn

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score

# X: (n_samples, n_features)  y: (n_samples,)
# Unlabelled rows are marked with -1
X, y = load_my_tabular_dataset()          # your helper
mask_unl = y == -1                        # boolean mask

X_scaled = StandardScaler().fit_transform(X)   # important for RBF

lp = LabelPropagation(kernel='knn',       # or 'rbf'
                      n_neighbors=10,     # ignored if kernel='rbf'
                      gamma=20,           # RBF bandwidth; ignored for knn
                      max_iter=1000, tol=1e-3)

lp.fit(X_scaled, y)

y_full = lp.transduction_                 # labels for *all* rows
y_unl_pred = y_full[mask_unl]

print("Propagated", mask_unl.sum(), "labels")
print("Accuracy on held-out labelled set:",
      accuracy_score(y_holdout_true, y_full[y_holdout_idx]))
```

Tip: for categorical columns, one-hot before scaling, or use a distance metric that respects category mismatches (e.g. Hamming + numeric Mahalanobis) and pass a callable kernel to LabelPropagation.

3.2 Hyperparameter heuristics

ParameterDefaultWhen to change
kernel='rbf'good for dense, low-dim dataswitch to 'knn' when features have mixed scales or p ≫ 30
gamma (RBF)20set to 1/(2\sigma^{2}). Cross-validate over log-space  [10^{-2},10^{3}].
n_neighbors (k-NN)7rule-of-thumb: \sqrt{n} for small n; decay to n^{1/3} for big n.
alpha (LabelSpreading only)0.2bigger = stronger clamping; 0.1–0.3 usually works.

3.3 Scaling to large tables
•Build the k-NN graph with FAISS or Annoy and hand it to scikit-learn via a sparse affinity matrix.
•Use the efficient LP solver of Fujiwara & Iruka 2014 to avoid the O(n^{3}) inverse  👤.
•Batch over disconnected components to fit into memory.

3.4 Packaging for XTYLearner

```python
# xtylearner/models/labelprop.py
from sklearn.semi_supervised import LabelPropagation
from .registry import register_model

@register_model("lp_knn")
class LP_KNN:
    def __init__(self, n_neighbors=10):
        self.clf = LabelPropagation(kernel="knn", n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.clf.fit(X, y)      # y uses -1 for unlabelled
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
```

Now a one-line call in your training script:

```python
model = models.build_model("lp_knn", n_neighbors=15).fit(X_train, y_train)
```

———

4  Further reading
•Zhu & Ghahramani, “Learning from Labeled and Unlabeled Data with Label Propagation”, CMU Tech Rep 2002.  👤
•Zhu, Ghahramani & Lafferty, “Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions”, ICML 2003.  👤
•Fujiwara & Iruka, “Efficient Label Propagation”, JMLR 2014.  👤
•scikit-learn LabelPropagation docs (examples, API).  👤

With those few dozen lines you have a rock-solid classical baseline that costs almost nothing to maintain yet often gives surprisingly strong numbers on UCI-style tables.
