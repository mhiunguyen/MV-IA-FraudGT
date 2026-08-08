from yacs.config import CfgNode as CN

from fraudGT.graphgym.register import register_config


@register_config('cfg_mvia')
def set_cfg_mvia(cfg):
    """Configuration for the MV-IA-FraudGT downstream head and loss."""
    cfg.mvia = CN()

    # Common latent size used by the graph and transaction views.
    cfg.mvia.dim_hidden = 64
    cfg.mvia.graph_layers = 2
    cfg.mvia.transaction_layers = 2
    cfg.mvia.classifier_layers = 2
    cfg.mvia.dropout = 0.2

    # A scalar gate is easier to interpret: alpha close to one favours the
    # graph view, while alpha close to zero favours the transaction view.
    cfg.mvia.gate_hidden = 64

    # Weighted focal-loss focusing parameter.
    cfg.mvia.focal_gamma = 2.0

    # Validation thresholds logged by the optional threshold-sweep utility.
    cfg.mvia.thresholds = [0.05, 0.10, 0.15, 0.20, 0.25,
                           0.30, 0.35, 0.40, 0.45, 0.50]
