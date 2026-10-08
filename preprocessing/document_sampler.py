# =============================================================================
# NDA Research Project - Document Sampler
# =============================================================================
# Module 2: Reproducible sampling of NDA documents from the Kleister-NDA
# repository for processing.
#
# Research Design Decisions:
# - Sampling is deterministic via a fixed random seed.
# - Selected document IDs are persisted to a manifest file so that the
#   exact same sample can be reproduced or extended later.
# - The sampler never modifies files in external_data/.
# =============================================================================

import json
import random
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def discover_documents(
    documents_dir: str,
    extensions: tuple = (".pdf",),
) -> List[Path]:
    """
    Discover all document files in the given directory.

    Args:
        documents_dir: Path to the directory containing NDA PDFs.
        extensions: File extensions to look for.

    Returns:
        Sorted list of Path objects for discovered documents.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    doc_path = Path(documents_dir)
    if not doc_path.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}"
        )

    files = sorted([
        f for f in doc_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])

    logger.info(f"Discovered {len(files)} documents in {documents_dir}")
    return files


def sample_documents(
    documents_dir: str,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> List[Path]:
    """
    Select a reproducible sample of NDA documents.

    If sample_size is None, -1, or >= total documents, all documents
    are returned (sorted deterministically).

    Args:
        documents_dir: Path to directory containing NDA PDFs.
        sample_size: Number of documents to sample. None/-1 = all.
        random_seed: Seed for reproducible sampling.

    Returns:
        List of selected document Paths.
    """
    all_docs = discover_documents(documents_dir)

    if sample_size is None or sample_size < 0 or sample_size >= len(all_docs):
        logger.info(f"Using all {len(all_docs)} documents (no sampling).")
        return all_docs

    # Sort first for determinism, then shuffle with seed
    doc_paths = sorted(all_docs, key=lambda p: p.name)
    rng = random.Random(random_seed)
    sampled = rng.sample(doc_paths, sample_size)
    # Re-sort for consistent ordering in output
    sampled.sort(key=lambda p: p.name)

    logger.info(
        f"Sampled {len(sampled)} documents from {len(all_docs)} "
        f"(seed={random_seed})"
    )
    return sampled


def save_sample_manifest(
    sampled_docs: List[Path],
    output_path: str,
    sample_size: int,
    random_seed: int,
) -> None:
    """
    Save the list of selected document IDs to a JSON manifest.

    This enables exact reproduction of the same sample in future runs.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "sample_size": sample_size,
        "random_seed": random_seed,
        "actual_count": len(sampled_docs),
        "document_ids": [p.stem for p in sampled_docs],
        "document_files": [p.name for p in sampled_docs],
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Sample manifest saved to {output_path}")


def load_sample_manifest(manifest_path: str) -> List[str]:
    """
    Load document IDs from a previously saved sample manifest.

    Returns:
        List of document ID strings (filename stems).
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["document_ids"]
