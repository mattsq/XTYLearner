from .generative import GenerativeTrainer


class GNNTrainer(GenerativeTrainer):
    """Generative trainer with extra CLI arguments for GNN-SCM."""

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--lambda_acyc", type=float, default=10.0)
        parser.add_argument("--gamma_l1", type=float, default=1e-2)
        return parser


__all__ = ["GNNTrainer"]
