from pathlib import Path
import shutil
import hashlib
import os
import pandas as pd


# -----------------------------
# Docker / environment settings
# -----------------------------

# Assumes the repo is mounted/copied at /workspace/ANSA
# and the command is run from the repo root.
METADATA_DIR = Path(os.getenv("METADATA_DIR", "/workspace/ANSA/metadata"))

CLINICAL_MANIFEST = Path(
    os.getenv("CLINICAL_MANIFEST", METADATA_DIR / "clinical_split_manifest.csv")
)

LOGARITHMIC_MANIFEST = Path(
    os.getenv("LOGARITHMIC_MANIFEST", METADATA_DIR / "logarithmic_split_manifest.csv")
)

# One unordered folder containing all unique .pt tensor files from OSF.
TENSOR_SOURCE_ROOT = Path(
    os.getenv("TENSOR_SOURCE_ROOT", "/workspace/data/tensor_files")
)

# Rebuilt organized output folders.
CLINICAL_OUTPUT_ROOT = Path(
    os.getenv("CLINICAL_OUTPUT_ROOT", "/workspace/rebuilt/reduced")
)

LOGARITHMIC_OUTPUT_ROOT = Path(
    os.getenv("LOGARITHMIC_OUTPUT_ROOT", "/workspace/rebuilt/reduced_log")
)

# Environment variable booleans:
#   OVERWRITE_OUTPUT=true
#   VERIFY_SHA256=false
OVERWRITE_OUTPUT = os.getenv("OVERWRITE_OUTPUT", "true").lower() == "true"
VERIFY_SHA256 = os.getenv("VERIFY_SHA256", "true").lower() == "true"


# -----------------------------
# Helper functions
# -----------------------------

def file_sha256(path, chunk_size=1024 * 1024):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)

    return sha.hexdigest()


def build_filename_index(source_root):
    """
    Recursively indexes .pt files in an unordered source folder.

    Returns:
        dict mapping filename -> source path

    Raises:
        ValueError if duplicate filenames are found.
    """
    source_root = Path(source_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Tensor source folder does not exist: {source_root}")

    index = {}
    duplicates = {}

    for path in source_root.rglob("*.pt"):
        filename = path.name

        if filename in index:
            duplicates.setdefault(filename, [index[filename]]).append(path)
        else:
            index[filename] = path

    if duplicates:
        message = ["Duplicate .pt filenames found in tensor source folder:"]
        for filename, paths in list(duplicates.items())[:20]:
            message.append(f"  {filename}")
            for p in paths:
                message.append(f"    {p}")

        if len(duplicates) > 20:
            message.append(f"  ... and {len(duplicates) - 20} more duplicate filenames")

        raise ValueError("\n".join(message))

    return index


def split_to_folder_name(split):
    """
    Converts manifest split names to dataset folder names.
    """
    mapping = {
        "train": "Training",
        "val": "Validation",
        "validation": "Validation",
        "test": "Testing",
    }

    split = str(split).strip().lower()

    if split not in mapping:
        raise ValueError(f"Unknown split value in manifest: {split}")

    return mapping[split]


def safe_label_folder(label_folder):
    """
    Keeps label folders exactly as stored in the manifest.

    Examples:
      high
      undetectable
      0.undetectable
      4.very high
    """
    return str(label_folder).strip()


def validate_manifest(manifest, manifest_path):
    required_columns = {
        "sample_id",
        "split",
        "label_folder",
        "filename",
    }

    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"{manifest_path} is missing required columns: {sorted(missing_columns)}"
        )

    duplicate_rows = manifest[
        manifest.duplicated(["split", "label_folder", "filename"], keep=False)
    ]

    if not duplicate_rows.empty:
        raise ValueError(
            f"{manifest_path} contains duplicate split/label/filename rows:\n"
            f"{duplicate_rows[['sample_id', 'split', 'label_folder', 'filename']]}"
        )

    duplicate_sample_ids = manifest[
        manifest.duplicated("sample_id", keep=False)
    ]

    if not duplicate_sample_ids.empty:
        raise ValueError(
            f"{manifest_path} contains duplicate sample_id values:\n"
            f"{duplicate_sample_ids[['sample_id', 'split', 'label_folder', 'filename']]}"
        )


def rebuild_one_manifest(
    manifest_path,
    filename_index,
    output_root,
    overwrite_output=False,
    verify_sha256=True,
):
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest, manifest_path)

    has_sha256 = "sha256" in manifest.columns

    if verify_sha256 and not has_sha256:
        print(f"Warning: {manifest_path} has no sha256 column. Skipping checksum verification.")
        verify_sha256 = False

    if overwrite_output and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []
    checksum_failures = []

    for _, row in manifest.iterrows():
        filename = str(row["filename"]).strip()
        split_folder = split_to_folder_name(row["split"])
        label_folder = safe_label_folder(row["label_folder"])

        source_path = filename_index.get(filename)

        if source_path is None:
            missing.append(filename)
            continue

        destination_path = output_root / split_folder / label_folder / filename
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, destination_path)
        copied += 1

        if verify_sha256:
            expected_sha = str(row["sha256"]).strip()
            observed_sha = file_sha256(destination_path)

            if observed_sha != expected_sha:
                checksum_failures.append(
                    {
                        "filename": filename,
                        "destination_path": str(destination_path),
                        "expected_sha256": expected_sha,
                        "observed_sha256": observed_sha,
                    }
                )

    print()
    print(f"Finished rebuilding from: {manifest_path}")
    print(f"Output root: {output_root}")
    print(f"Copied files: {copied}")
    print(f"Missing source files: {len(missing)}")
    print(f"Checksum failures: {len(checksum_failures)}")

    if missing:
        print()
        print("Missing source files, first 20:")
        for filename in missing[:20]:
            print(f"  {filename}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if checksum_failures:
        print()
        print("Checksum failures, first 20:")
        for failure in checksum_failures[:20]:
            print(f"  {failure['filename']}")
            print(f"    destination: {failure['destination_path']}")
            print(f"    expected:    {failure['expected_sha256']}")
            print(f"    observed:    {failure['observed_sha256']}")
        if len(checksum_failures) > 20:
            print(f"  ... and {len(checksum_failures) - 20} more")

    if missing or checksum_failures:
        raise RuntimeError(
            f"Rebuild from {manifest_path} completed with errors. See messages above."
        )

    return copied


def main():
    print("ANSA data rebuilder")
    print("-------------------")
    print(f"Metadata directory: {METADATA_DIR}")
    print(f"Clinical manifest: {CLINICAL_MANIFEST}")
    print(f"Logarithmic manifest: {LOGARITHMIC_MANIFEST}")
    print(f"Tensor source root: {TENSOR_SOURCE_ROOT}")
    print(f"Clinical output root: {CLINICAL_OUTPUT_ROOT}")
    print(f"Logarithmic output root: {LOGARITHMIC_OUTPUT_ROOT}")
    print(f"Overwrite output: {OVERWRITE_OUTPUT}")
    print(f"Verify SHA256: {VERIFY_SHA256}")

    print()
    print("Indexing unordered tensor source folder...")
    filename_index = build_filename_index(TENSOR_SOURCE_ROOT)
    print(f"Indexed .pt files: {len(filename_index)}")

    clinical_copied = rebuild_one_manifest(
        manifest_path=CLINICAL_MANIFEST,
        filename_index=filename_index,
        output_root=CLINICAL_OUTPUT_ROOT,
        overwrite_output=OVERWRITE_OUTPUT,
        verify_sha256=VERIFY_SHA256,
    )

    logarithmic_copied = rebuild_one_manifest(
        manifest_path=LOGARITHMIC_MANIFEST,
        filename_index=filename_index,
        output_root=LOGARITHMIC_OUTPUT_ROOT,
        overwrite_output=OVERWRITE_OUTPUT,
        verify_sha256=VERIFY_SHA256,
    )

    print()
    print("Rebuild complete.")
    print(f"Clinical files copied: {clinical_copied}")
    print(f"Logarithmic files copied: {logarithmic_copied}")
    print(f"Clinical rebuilt folder: {CLINICAL_OUTPUT_ROOT}")
    print(f"Logarithmic rebuilt folder: {LOGARITHMIC_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()