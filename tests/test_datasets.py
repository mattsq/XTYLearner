from xtylearner.data import load_toy_dataset, load_synthetic_dataset


def test_load_toy_dataset_shapes():
    ds = load_toy_dataset(n_samples=10, d_x=3, seed=1)
    X, Y, T = ds.tensors
    assert X.shape == (10, 3)
    assert Y.shape == (10, 1)
    assert T.shape == (10,)


def test_load_synthetic_dataset_shapes():
    ds = load_synthetic_dataset(n_samples=8, d_x=4, seed=2)
    X, Y, T = ds.tensors
    assert X.shape == (8, 4)
    assert Y.shape == (8, 1)
    assert T.shape == (8,)
