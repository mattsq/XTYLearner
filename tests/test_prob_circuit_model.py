import numpy as np
import pandas as pd

from xtylearner.data import load_toy_dataset
from xtylearner.models.prob_circuit_model import ProbCircuitModel


def test_prob_circuit_model_fit_and_predict():
    ds = load_toy_dataset(n_samples=50, d_x=2, seed=0)
    X, Y, T = ds.tensors
    X = X.numpy()
    Y = Y.numpy().reshape(-1)
    T = T.numpy()

    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["T"] = T
    df["Y"] = Y

    rng = np.random.default_rng(0)
    mask = rng.random(len(df)) < 0.3
    df.loc[mask, "T"] = np.nan

    model = ProbCircuitModel(min_instances_slice=20)
    model.fit(df)

    probs = model.predict_t_posterior(X, Y)
    assert probs.shape == (50,)
    assert np.all((probs >= 0) & (probs <= 1))
