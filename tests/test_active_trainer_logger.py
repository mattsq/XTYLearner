import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_mixed_synthetic_dataset
from xtylearner.models import MultiTask
from xtylearner.training import ActiveTrainer, ConsoleLogger
from xtylearner.active import QueryStrategy

class DummyStrategy(QueryStrategy):
    def forward(self, model, X_unlab, rep_fn, batch_size):
        return torch.zeros(len(X_unlab))

def test_active_trainer_logs_progress(capsys):
    ds = load_mixed_synthetic_dataset(n_samples=20, d_x=2, seed=0, label_ratio=0.5)
    loader = DataLoader(ds, batch_size=4)
    model = MultiTask(d_x=2, d_y=1, k=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = ActiveTrainer(
        model,
        opt,
        loader,
        DummyStrategy(),
        budget=2,
        batch=1,
        trainer_logger=ConsoleLogger(print_every=1),
    )
    trainer.fit(1)
    out = capsys.readouterr().out
    assert "labelled=" in out
    assert "budget=" in out
