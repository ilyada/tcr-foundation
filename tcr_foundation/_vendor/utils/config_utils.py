"""
Configuration management utilities for unified TCR-BERT training.
"""

import os
import yaml
from pathlib import Path
import argparse


# ================= CONFIG LOADING =================

def load_config(yaml_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        yaml_path: Absolute or relative path to YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    config_file = Path(yaml_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Merge command-line argument overrides into configuration dictionary.

    Args:
        config: Base configuration dictionary from YAML
        args: Parsed command-line arguments

    Returns:
        Updated configuration dictionary with CLI overrides applied

    Priority: CLI arguments override YAML config values
    """
    # Direct overrides
    if args.output_dir is not None:
        config["save_path"] = args.output_dir

    if args.pretrained_model is not None:
        if args.mode == "finetune":
            config["model_name"] = args.pretrained_model
        elif args.mode == "pretrain":
            config["model_path"] = args.pretrained_model
        elif args.mode == "sequential":
            # For sequential mode, override the finetune model_name
            # (pretrain output will override this later)
            if "finetune" in config:
                config["finetune"]["model_name"] = args.pretrained_model

    # Granular epoch overrides (take precedence over --epochs shorthand)
    if args.mode == "pretrain":
        # Pretrain mode: only pretrain epochs matter
        if hasattr(args, 'pretrain_epochs') and args.pretrain_epochs is not None:
            config["epochs"] = args.pretrain_epochs
            if "pretrain" in config:
                config["pretrain"]["epochs"] = args.pretrain_epochs
        elif args.epochs is not None:
            config["epochs"] = args.epochs
            if "pretrain" in config:
                config["pretrain"]["epochs"] = args.epochs

    elif args.mode == "finetune":
        # Finetune mode: warmup and finetune epochs
        if hasattr(args, 'warmup_epochs') and args.warmup_epochs is not None:
            config["warm_up_epochs"] = args.warmup_epochs
            if "finetune" in config:
                config["finetune"]["warm_up_epochs"] = args.warmup_epochs
        if hasattr(args, 'finetune_epochs') and args.finetune_epochs is not None:
            config["fine_tune_epochs"] = args.finetune_epochs
            if "finetune" in config:
                config["finetune"]["fine_tune_epochs"] = args.finetune_epochs
        elif args.epochs is not None:
            config["fine_tune_epochs"] = args.epochs
            if "finetune" in config:
                config["finetune"]["fine_tune_epochs"] = args.epochs

    elif args.mode == "sequential":
        # Sequential mode: all three epoch types
        if "pretrain" in config:
            if hasattr(args, 'pretrain_epochs') and args.pretrain_epochs is not None:
                config["pretrain"]["epochs"] = args.pretrain_epochs
            elif args.epochs is not None:
                # Fallback: --epochs sets pretrain epochs
                config["pretrain"]["epochs"] = args.epochs

        if "finetune" in config:
            if hasattr(args, 'warmup_epochs') and args.warmup_epochs is not None:
                config["finetune"]["warm_up_epochs"] = args.warmup_epochs
            if hasattr(args, 'finetune_epochs') and args.finetune_epochs is not None:
                config["finetune"]["fine_tune_epochs"] = args.finetune_epochs
            elif args.epochs is not None:
                # Fallback: --epochs sets finetune epochs (not warmup)
                config["finetune"]["fine_tune_epochs"] = args.epochs

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
        if args.mode in ("pretrain", "sequential") and "pretrain" in config:
            config["pretrain"]["batch_size"] = args.batch_size
        if args.mode in ("finetune", "sequential") and "finetune" in config:
            config["finetune"]["batch_size"] = args.batch_size

    if args.lr is not None:
        if args.mode == "pretrain":
            config["lr"] = args.lr
            if "pretrain" in config:
                config["pretrain"]["lr"] = args.lr
        elif args.mode == "finetune":
            config["lr_bert"] = args.lr
            if "finetune" in config:
                config["finetune"]["lr_bert"] = args.lr
        elif args.mode == "sequential":
            if "pretrain" in config:
                config["pretrain"]["lr"] = args.lr
            if "finetune" in config:
                config["finetune"]["lr_bert"] = args.lr

    if args.seed is not None:
        config["seed"] = args.seed
        if args.mode in ("pretrain", "sequential") and "pretrain" in config:
            config["pretrain"]["seed"] = args.seed
        if args.mode in ("finetune", "sequential") and "finetune" in config:
            config["finetune"]["seed"] = args.seed

    # Scheduler override
    if hasattr(args, 'scheduler') and args.scheduler is not None:
        config["scheduler"] = args.scheduler
        if args.mode == "sequential" and "finetune" in config:
            config["finetune"]["scheduler"] = args.scheduler

    # Pooling override
    if hasattr(args, 'pooling') and args.pooling is not None:
        config["pooling"] = args.pooling
        if args.mode == "sequential" and "finetune" in config:
            config["finetune"]["pooling"] = args.pooling

    # Head architecture overrides
    if hasattr(args, 'head_layers') and args.head_layers is not None:
        config["head_layers"] = args.head_layers
        if args.mode == "sequential" and "finetune" in config:
            config["finetune"]["head_layers"] = args.head_layers

    if hasattr(args, 'head_hidden') and args.head_hidden is not None:
        # Parse comma-separated string to list of ints
        head_hidden = [int(x.strip()) for x in args.head_hidden.split(",")]
        config["head_hidden"] = head_hidden
        if args.mode == "sequential" and "finetune" in config:
            config["finetune"]["head_hidden"] = head_hidden

    # Save path override
    if hasattr(args, 'save_path') and args.save_path is not None:
        config["save_path"] = args.save_path
        if args.mode == "pretrain" and "pretrain" in config:
            config["pretrain"]["save_path"] = args.save_path
        elif args.mode == "sequential" and "finetune" in config:
            config["finetune"]["save_path"] = args.save_path

    # Data path override
    if hasattr(args, 'data_path') and args.data_path is not None:
        config["data_path"] = args.data_path
        if args.mode == "sequential":
            if "pretrain" in config:
                config["pretrain"]["data_path"] = args.data_path
            if "finetune" in config:
                config["finetune"]["data_path"] = args.data_path

    # Model name override
    if hasattr(args, 'model_name') and args.model_name is not None:
        config["model_name"] = args.model_name
        if args.mode == "pretrain":
            config["model_path"] = args.model_name
            if "pretrain" in config:
                config["pretrain"]["model_path"] = args.model_name
        elif args.mode == "sequential" and "pretrain" in config:
            config["pretrain"]["model_path"] = args.model_name

    # Multi-task learning overrides
    if hasattr(args, 'property_prediction') and args.property_prediction:
        config["property_prediction"] = True
        if args.mode in ("pretrain", "sequential") and "pretrain" in config:
            config["pretrain"]["property_prediction"] = True
    if hasattr(args, 'no_property_prediction') and args.no_property_prediction:
        config["property_prediction"] = False
        if args.mode in ("pretrain", "sequential") and "pretrain" in config:
            config["pretrain"]["property_prediction"] = False

    if hasattr(args, 'property_alpha') and args.property_alpha is not None:
        config["property_alpha"] = args.property_alpha
        if args.mode in ("pretrain", "sequential") and "pretrain" in config:
            config["pretrain"]["property_alpha"] = args.property_alpha

    if hasattr(args, 'prototype_learning') and args.prototype_learning:
        config["prototype_learning"] = True
        if args.mode in ("pretrain", "sequential") and "pretrain" in config:
            config["pretrain"]["prototype_learning"] = True
    if hasattr(args, 'no_prototype_learning') and args.no_prototype_learning:
        config["prototype_learning"] = False
        if args.mode in ("pretrain", "sequential") and "pretrain" in config:
            config["pretrain"]["prototype_learning"] = False

    if hasattr(args, 'prototype_temperature') and args.prototype_temperature is not None:
        config["prototype_temperature"] = args.prototype_temperature
        if args.mode in ("pretrain", "sequential") and "pretrain" in config:
            config["pretrain"]["prototype_temperature"] = args.prototype_temperature

    # Regularization overrides
    if hasattr(args, 'dropout') and args.dropout is not None:
        config["dropout"] = args.dropout
        if args.mode == "sequential" and "finetune" in config:
            config["finetune"]["dropout"] = args.dropout

    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
        if args.mode in ("pretrain", "sequential") and "pretrain" in config:
            config["pretrain"]["weight_decay"] = args.weight_decay
        if args.mode in ("finetune", "sequential") and "finetune" in config:
            config["finetune"]["weight_decay"] = args.weight_decay

    return config


# ================= CONFIG VALIDATION =================

def validate_config(config: dict, mode: str) -> None:
    """
    Validate configuration has all required fields for the specified mode.

    Args:
        config: Configuration dictionary to validate
        mode: Training mode ('pretrain', 'finetune', or 'sequential')

    Raises:
        ValueError: If required fields are missing or invalid
    """
    if mode == "pretrain":
        required = [
            "model_path", "save_path", "data_path", "epochs",
            "batch_size", "lr", "max_len", "mlm_prob"
        ]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field for pretrain mode: {field}")

    elif mode == "finetune":
        # Check if using CDR2+CDR3 mode (same data file, split by condition) or CDR3-only mode (separate files)
        use_cdr2_cdr3 = config.get("use_cdr2_cdr3", False)

        # Common required fields
        required = [
            "model_name", "save_path", "warm_up_epochs",
            "fine_tune_epochs", "batch_size", "max_len", "lr_head_warm_up",
            "lr_bert", "lr_head", "dropout", "label_smoothing"
        ]

        # Add data field based on mode
        if use_cdr2_cdr3:
            required.append("data_path")  # CDR2+CDR3: same file as pretrain, split by condition
        else:
            required.append("data_dir")   # Legacy CDR3-only: separate files per split

        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field for finetune mode: {field}")

    elif mode == "sequential":
        if "pretrain" not in config or "finetune" not in config:
            raise ValueError("Sequential mode requires 'pretrain' and 'finetune' sections")
        validate_config(config["pretrain"], "pretrain")
        validate_config(config["finetune"], "finetune")

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'pretrain', 'finetune', or 'sequential'")


# ================= COMET ML SETUP =================

def setup_comet_experiment(config: dict, args: argparse.Namespace):
    """
    Initialize Comet ML experiment for logging.

    Args:
        config: Configuration dictionary (must contain comet_project, comet_workspace)
        args: CLI arguments (for experiment_name override and no_comet flag)

    Returns:
        Initialized Comet ML Experiment object, or None if --no-comet flag set

    Environment Variables:
        COMET_API_KEY: API key for Comet ML (required if not in args)
    """
    if args.no_comet:
        return None

    try:
        from comet_ml import Experiment
    except ImportError:
        print("WARNING: comet_ml not installed. Continuing without logging.")
        print("Install with: pip install comet-ml")
        return None

    # Get API key
    api_key = getattr(args, 'comet_api_key', None) or os.getenv("COMET_API_KEY")
    if not api_key:
        print("WARNING: Comet ML API key not found. Set COMET_API_KEY environment variable")
        print("or use --comet-api-key argument. Continuing without logging.")
        return None

    # Create experiment
    try:
        exp = Experiment(
            api_key=api_key,
            project_name=getattr(args, 'comet_project', None) or config.get("comet_project", "tcr-bert"),
            workspace=config.get("comet_workspace", "ilyada")
        )

        # Set experiment name and tags
        if hasattr(args, 'experiment_name') and args.experiment_name:
            exp.set_name(args.experiment_name)
        elif "experiment_name" in config:
            exp.set_name(config["experiment_name"])

        if "tags" in config:
            exp.add_tags(config["tags"])

        # Log config
        exp.log_parameters(config)

        return exp

    except Exception as e:
        print(f"WARNING: Comet ML initialization failed: {e}")
        print("Continuing without logging.")
        return None
