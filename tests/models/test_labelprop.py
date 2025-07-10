from sklearn.linear_model import LinearRegression

from xtylearner.data import load_mixed_synthetic_dataset, load_synthetic_dataset
from xtylearner.models import LP_KNN


def test_lp_knn_regressor_metrics():
    train = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=0, label_ratio=0.7)
    val = load_synthetic_dataset(n_samples=10, d_x=2, seed=1)
    Xtr, Ytr, Ttr = train.tensors
    Xv, Yv, Tv = val.tensors
    model = LP_KNN(n_neighbors=3, regressor=LinearRegression())
    model.fit(Xtr.numpy(), Ytr.squeeze(-1).numpy(), Ttr.numpy())
    metrics = model.regressor_metrics(
        Xtr.numpy(), Ytr.squeeze(-1).numpy(), Ttr.numpy(),
        Xv.numpy(), Yv.squeeze(-1).numpy(), Tv.numpy()
    )
    assert set(metrics) == {"val_acc", "rmse_labelled", "rmse_full"}
    assert all(isinstance(v, float) for v in metrics.values())
