"""Train DeconfounderCFM on the Twins dataset and report PEHE."""

import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_twins
from xtylearner.models import DeconfounderCFM
from xtylearner.training import Trainer, ConsoleLogger


def main() -> None:
    ds = load_twins()
    X, Y, T = ds.tensors
    T = T.float().unsqueeze(1)
    loader = DataLoader(
        torch.utils.data.TensorDataset(X, Y, T), batch_size=512, shuffle=True
    )
    model = DeconfounderCFM(d_x=X.size(1), d_y=1, k_t=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger = ConsoleLogger()
    trainer = Trainer(model, opt, loader, logger=logger)
    trainer.fit(2)

    with torch.no_grad():
        y0 = model.predict_outcome(X, torch.zeros_like(T))
        y1 = model.predict_outcome(X, torch.ones_like(T))
    ite_pred = (y1 - y0).squeeze(1)
    print("PEHE", torch.sqrt(((ite_pred - 2.0) ** 2).mean()).item())


if __name__ == "__main__":
    main()
