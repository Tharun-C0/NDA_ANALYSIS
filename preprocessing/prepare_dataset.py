# =============================================================================
# NDA Research Project - Main Dataset Preparation Pipeline
# =============================================================================
# Module 1: End-to-End Dataset Preparation
#
# This script orchestrates the full Module 1 pipeline:
#   1. Load configuration
#   2. Load raw NDA documents
#   3. Preprocess clause text
#   4. Compute and display dataset statistics
#   5. Analyze class imbalance
#   6. Split into train/val/test sets
#   7. Save all outputs
#
# Usage:
#   python preprocessing/prepare_dataset.py
#   python preprocessing/prepare_dataset.py --config path/to/config.yaml
#
# =============================================================================

import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path so imports work when running as script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from configs.config_loader import load_config
from preprocessing.dataset_loader import load_dataset
from preprocessing.text_preprocessor import preprocess_clauses
from preprocessing.dataset_statistics import (
    compute_statistics,
    print_statistics,
    save_statistics_json,
    save_class_frequencies_csv,
)
from preprocessing.dataset_splitter import (
    split_by_documents,
    save_splits,
    save_split,
)


def setup_logging(config: dict) -> None:
    """Configure logging based on config settings."""
    log_level = config.get("logging", {}).get("level", "INFO")
    log_to_file = config.get("logging", {}).get("log_to_file", False)
    log_dir = config.get("logging", {}).get("log_dir", "reports/logs")
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / "prepare_dataset.log",
            mode="w",
            encoding="utf-8",
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def main(config_path: str = None) -> None:
    """
    Run the complete Module 1 dataset preparation pipeline.
    
    Args:
        config_path: Optional path to config.yaml override.
    """
    # -------------------------------------------------------------------------
    # Step 0: Load configuration
    # -------------------------------------------------------------------------
    config = load_config(config_path)
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("NDA Research Project - Module 1: Dataset Preparation")
    logger.info("=" * 70)
    
    raw_dir = config["dataset"]["raw_dir"]
    processed_dir = config["dataset"]["processed_dir"]
    splits_dir = config["dataset"]["splits_dir"]
    reports_dir = config["output"]["reports_dir"]
    
    seed = config.get("random_seed", 42)
    save_format = config.get("output", {}).get("save_format", "json")
    
    preprocessing_cfg = config.get("preprocessing", {})
    splitting_cfg = config.get("splitting", {})
    parsing_cfg = config.get("parsing", {})
    
    # -------------------------------------------------------------------------
    # Step 1: Load raw NDA documents
    # -------------------------------------------------------------------------
    logger.info("Step 1: Loading raw NDA documents...")
    
    parse_kwargs = {}
    if parsing_cfg:
        tag_mapping = {
            "clause_start_tag": "clause_start",
            "clause_end_tag": "clause_end",
            "class_start_tag": "class_start",
            "class_end_tag": "class_end",
        }
        for config_key, func_key in tag_mapping.items():
            if config_key in parsing_cfg:
                parse_kwargs[func_key] = parsing_cfg[config_key]
    
    clauses, warnings = load_dataset(
        raw_dir=raw_dir,
        encoding=preprocessing_cfg.get("file_encoding", "utf-8"),
        **parse_kwargs,
    )
    
    if not clauses:
        logger.error(
            "No clauses were loaded. Please check that:\n"
            f"  1. NDA .txt files exist in: {raw_dir}\n"
            "  2. Files use the expected annotation format:\n"
            "     [INIT_CLAUSE]\n"
            "     clause text\n"
            "     [INIT_CLASSE]labels[END_CLASSE]\n"
            "     [END_CLAUSE]\n"
        )
        sys.exit(1)
    
    logger.info(f"Loaded {len(clauses)} clauses from raw documents.")
    
    # Print any loading warnings
    if warnings:
        logger.info(f"\n  Warnings during loading ({sum(len(w) for w in warnings.values())} total):")
        for doc_id, doc_warnings in warnings.items():
            for w in doc_warnings:
                logger.warning(f"  {w}")
    
    # -------------------------------------------------------------------------
    # Step 2: Preprocess clauses
    # -------------------------------------------------------------------------
    logger.info("Step 2: Preprocessing clause text...")
    
    processed_clauses = preprocess_clauses(
        clauses,
        strip_ws=preprocessing_cfg.get("strip_whitespace", True),
        do_normalize_unicode=preprocessing_cfg.get("normalize_unicode", True),
    )
    
    logger.info(f"After preprocessing: {len(processed_clauses)} clauses retained.")
    
    # Save processed dataset
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    ext = "json" if save_format == "json" else "csv"
    processed_file = processed_path / f"all_clauses.{ext}"
    
    from preprocessing.dataset_splitter import save_split
    save_split(processed_clauses, str(processed_file), save_format)
    logger.info(f"Processed dataset saved to: {processed_file}")
    
    # -------------------------------------------------------------------------
    # Step 3: Compute and display dataset statistics
    # -------------------------------------------------------------------------
    logger.info("Step 3: Computing dataset statistics...")
    
    stats = compute_statistics(processed_clauses)
    print_statistics(stats)
    
    # Save statistics
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    save_statistics_json(stats, str(reports_path / "dataset_statistics.json"))
    
    # -------------------------------------------------------------------------
    # Step 4: Class imbalance analysis
    # -------------------------------------------------------------------------
    logger.info("Step 4: Analyzing class imbalance...")
    
    save_class_frequencies_csv(stats, str(reports_path / "class_frequencies.csv"))
    
    logger.info(
        f"Majority classes: {', '.join(stats['labels']['majority_classes'])}"
    )
    logger.info(
        f"Minority classes: {', '.join(stats['labels']['minority_classes'])}"
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Create train/val/test splits
    # -------------------------------------------------------------------------
    logger.info("Step 5: Creating train/val/test splits...")
    
    train, val, test = split_by_documents(
        processed_clauses,
        train_ratio=splitting_cfg.get("train_ratio", 0.7),
        val_ratio=splitting_cfg.get("val_ratio", 0.15),
        test_ratio=splitting_cfg.get("test_ratio", 0.15),
        random_seed=seed,
    )
    
    # Save splits
    save_splits(train, val, test, splits_dir, save_format)
    
    # Print split summary
    print("\n" + "=" * 70)
    print("  DATASET SPLITS (document-level, no data leakage)")
    print("=" * 70)
    
    for name, split_data in [("Train", train), ("Val", val), ("Test", test)]:
        n_docs = len(set(c.doc_id for c in split_data))
        n_clauses = len(split_data)
        print(f"  {name + ':':<10} {n_docs:>4} documents, {n_clauses:>5} clauses")
    
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 6: Summary
    # -------------------------------------------------------------------------
    print("\n  ✅ Module 1 complete. Outputs saved to:")
    print(f"     Processed data : {processed_dir}")
    print(f"     Dataset splits : {splits_dir}")
    print(f"     Statistics     : {reports_dir}/dataset_statistics.json")
    print(f"     Class freq CSV : {reports_dir}/class_frequencies.csv")
    print(f"     Log file       : {config.get('logging', {}).get('log_dir', 'reports/logs')}/prepare_dataset.log")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NDA Research Project - Module 1: Dataset Preparation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
