"""Train CEVAE-M on the IHDP semi-synthetic dataset."""

import torch
from torch.utils.data import DataLoader

from xtylearner.data import load_ihdp
from xtylearner.models import CEVAE_M
from xtylearner.training import Trainer, ConsoleLogger


def main() -> None:
    ds = load_ihdp()
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    model = CEVAE_M(d_x=ds.tensors[0].size(1), d_y=1, k=2, d_z=32, hidden=128, tau=0.66)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger = ConsoleLogger()
    trainer = Trainer(model, opt, loader, logger=logger)
    trainer.fit(1)


if __name__ == "__main__":
    main()
