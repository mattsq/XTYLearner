import argparse
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader

from xtylearner.data import get_dataset
from xtylearner.models import get_model
import xtylearner.training as training


DEFAULT_CFG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XTYLearner model")
    parser.add_argument(
        "--model", required=True, help="Model name registered in xtylearner.models"
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name from xtylearner.data"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else DEFAULT_CFG
    cfg = load_config(cfg_path)

    dataset_cfg = cfg.get("dataset", {})
    dataset_name = args.dataset if args.dataset else dataset_cfg.get("name", "toy")
    if dataset_name == dataset_cfg.get("name"):
        dataset_params = dataset_cfg.get("params", {})
    else:
        dataset_params = {}
    dataset = get_dataset(dataset_name, **dataset_params)

    model_cfg = cfg.get("model", {})
    model_name = args.model if args.model else model_cfg.get("name")
    if model_name == model_cfg.get("name"):
        model_params = model_cfg.get("params", {})
    else:
        model_params = {}
    model = get_model(model_name, **model_params)

    train_cfg = cfg.get("training", {})
    trainer_name = train_cfg.get("trainer", "SupervisedTrainer")
    batch_size = train_cfg.get("batch_size", 32)
    learning_rate = train_cfg.get("learning_rate", 1e-3)
    epochs = train_cfg.get("epochs", 1)

    trainer_cls = getattr(training, trainer_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = trainer_cls(model, optimizer, loader)
    trainer.fit(epochs)


if __name__ == "__main__":
    main()
