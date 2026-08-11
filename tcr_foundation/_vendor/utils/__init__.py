"""
Utility modules for unified TCR-BERT training script.
"""

from .data_utils import (
    TCRPairMLMDataset,
    TCRDataset,
    TCRPairDataset,
    load_and_filter_sequences,
    load_split,
    load_split_by_condition,
    leave_unique
)

from .model_utils import (
    TCRBertClassifier,
    init_mlm_model,
    init_classifier,
    freeze_bert_layers,
    unfreeze_last_n_layers,
    save_model_checkpoint
)

from .training_utils import (
    train_mlm_epoch,
    train_classifier_epoch,
    evaluate_classifier,
    predict_probs
)

from .config_utils import (
    load_config,
    merge_cli_overrides,
    validate_config,
    setup_comet_experiment
)

__all__ = [
    # Data utils
    "TCRPairMLMDataset",
    "TCRDataset",
    "TCRPairDataset",
    "load_and_filter_sequences",
    "load_split",
    "load_split_by_condition",
    "leave_unique",
    # Model utils
    "TCRBertClassifier",
    "init_mlm_model",
    "init_classifier",
    "freeze_bert_layers",
    "unfreeze_last_n_layers",
    "save_model_checkpoint",
    # Training utils
    "train_mlm_epoch",
    "train_classifier_epoch",
    "evaluate_classifier",
    "predict_probs",
    # Config utils
    "load_config",
    "merge_cli_overrides",
    "validate_config",
    "setup_comet_experiment",
]
