# ANSA ResNet Classification

<img width="2053" height="2071" alt="overview" src="https://github.com/user-attachments/assets/672b6489-4bde-4bdd-8d83-b0fd93867aca" />

This repository contains code and notebooks for semi-quantitative classification of HIV-1 nucleic acid amplification reactions using time-resolved fluorescence images from Amplification Nucleation Site Analysis (ANSA). ANSA produces spatially localized fluorescent amplification sites during isothermal amplification reactions in a microfluidic chip. At high target concentrations, these sites can overlap, making direct site counting unreliable. This project uses modified ResNet models to learn spatiotemporal fluorescence patterns from multi-frame image tensors and classify reactions into clinically relevant or logarithmically spaced DNA concentration ranges.

The analysis supports two classification tasks:

1. **Clinical model**: four classes based on clinically motivated HIV-1 input copy number ranges.
2. **Logarithmic model**: five classes spanning approximately five orders of magnitude.

In the associated manuscript draft, the clinical model achieved 94.6% test accuracy and the logarithmic model achieved 92.7% test accuracy, with most errors occurring between adjacent concentration classes.

## Repository structure

```text
ANSA/
├── Dockerfile
├── README.md
├── requirements.txt
├── notebooks/
│   ├── Clinical_model.ipynb
│   ├── Logarithmic_model.ipynb
│   ├── Optuna_clinical_model.ipynb
│   └── Optuna_logarithmic_model.ipynb
├── scripts/
│   └── rebuild_data_from_manifest.py
└── src/
    └── ansa/
        ├── data_utils.py
        └── visualization.py
```

The dataset is intentionally kept out of version control. The processed tensor data should be downloaded separately from OSF and mounted into the Docker container at runtime. Local `models/` and `results/` folders can be created wherever you want to save outputs.

Optional local output folders, not tracked by git:

```text
ANSA/
├── models/
└── results/
```

## Data availability

The raw fluorescence image stacks are not included in this repository. The OSF dataset is expected to contain the processed reduced tensor files and metadata manifests needed to rebuild the train, validation, and test folder structure used by the notebooks.

Expected OSF folder contents after download and extraction:

```text
ANSA_OSF_data/
├── Metadata/
│   ├── clinical_split_manifest.csv
│   ├── logarithmic_split_manifest.csv
│   └── sample_manifest.csv
└── processed_reduced_tensors/
    ├── *.pt
    └── ...
```

The `.pt` tensor files are stored in one unordered folder. The manifest files define how the same tensor files should be assigned to the clinical and logarithmic datasets.

After rebuilding, the mounted OSF data folder should look like this. The folder may be named `data/`, `ANSA_OSF_data/`, or any other local path on your computer, as long as it is mounted to `/home/jovyan/work/data` inside Docker.

```text
ANSA_OSF_data/
├── Metadata/
├── processed_reduced_tensors/
├── reduced/
│   ├── Training/
│   ├── Validation/
│   └── Testing/
└── reduced_log/
    ├── Training/
    ├── Validation/
    └── Testing/
```

where:

- `<mounted OSF folder>/reduced/` is used by the clinical model.
- `<mounted OSF folder>/reduced_log/` is used by the logarithmic model.

## Option 1: Run with Docker

Docker is the recommended way to reproduce the analysis environment. The Docker image contains the software environment, notebooks, source code, and rebuild script. The image does **not** contain the ANSA dataset. Download the dataset separately from OSF and mount it into the container at runtime.

There are two Docker options:

- **Option 1A: Pull the prebuilt image** from GitHub Container Registry. This is the easiest option for users who want to run the analysis without building the image themselves.
- **Option 1B: Build the image locally** from the Dockerfile. This is useful for development or if you want to modify the environment.

### Step 1A. Pull the prebuilt Docker image

A prebuilt Docker image is available from GitHub Container Registry:

```bash
docker pull ghcr.io/martc53/ansa-ml:latest
```

Use this image name in the Docker run commands below:

```text
ghcr.io/martc53/ansa-ml:latest
```

If the package is private, users must first authenticate with GitHub Container Registry using a GitHub account that has access to the package:

```bash
docker login ghcr.io
```

For public release, set the GitHub package visibility to public so users can pull the image without authentication.

### Step 1B. Or build the image locally

From the repository root:

```bash
docker build -t ansa-ml .
```

If you build locally, use this image name in the Docker run commands below:

```text
ansa-ml
```

### Step 2. Download the OSF data

Download and extract the OSF dataset to any convenient location on your computer. The OSF data folder does **not** need to be inside the GitHub repository.

For example, the local OSF folder could be one of the following:

Ubuntu/Linux:

```text
/home/username/ANSA_OSF_data
```

Windows PowerShell:

```text
C:\Users\username\Downloads\ANSA_OSF_data
```

After extraction, the OSF folder should contain:

```text
ANSA_OSF_data/
├── Metadata/
│   ├── clinical_split_manifest.csv
│   ├── logarithmic_split_manifest.csv
│   └── sample_manifest.csv
└── processed_reduced_tensors/
    ├── *.pt
    └── ...
```

### Step 3. Choose local output folders

Create local folders for trained models and generated results. These can be inside the repository or somewhere else on your computer.

Ubuntu/Linux, from the repository root:

```bash
mkdir -p models results
```

Windows PowerShell, from the repository root:

```powershell
mkdir models
mkdir results
```

You may also create a local `data/` folder in the repository and place the OSF contents there, but this is optional. The important point is that the OSF data folder you choose must be mounted to this path inside Docker:

```text
/home/jovyan/work/data
```

The Docker run commands below use placeholders for the local OSF folder. Replace them with the actual path on your computer. The commands use the prebuilt image by default:

```text
ghcr.io/martc53/ansa-ml:latest
```

If you built the image locally instead, replace `ghcr.io/martc53/ansa-ml:latest` with `ansa-ml`.

### Step 4A. Start Docker without a GPU

Use this command if Docker cannot access an NVIDIA GPU, or if you want to run CPU-only. This is sufficient for opening the notebooks, rebuilding the dataset, and testing the workflow. Training and Optuna optimization will be slower on CPU.

Ubuntu/Linux:

```bash
docker run -p 8888:8888 \
  -v /path/to/ANSA_OSF_data:/home/jovyan/work/data \
  -v "$(pwd)/models:/home/jovyan/work/models" \
  -v "$(pwd)/results:/home/jovyan/work/results" \
  ghcr.io/martc53/ansa-ml:latest
```

Example:

```bash
docker run -p 8888:8888 \
  -v /home/username/ANSA_OSF_data:/home/jovyan/work/data \
  -v "$(pwd)/models:/home/jovyan/work/models" \
  -v "$(pwd)/results:/home/jovyan/work/results" \
  ghcr.io/martc53/ansa-ml:latest
```

Windows PowerShell:

```powershell
docker run -p 8888:8888 `
  -v "C:\path\to\ANSA_OSF_data:/home/jovyan/work/data" `
  -v "${PWD}\models:/home/jovyan/work/models" `
  -v "${PWD}\results:/home/jovyan/work/results" `
  ghcr.io/martc53/ansa-ml:latest
```

Example:

```powershell
docker run -p 8888:8888 `
  -v "C:\Users\username\Downloads\ANSA_OSF_data:/home/jovyan/work/data" `
  -v "${PWD}\models:/home/jovyan/work/models" `
  -v "${PWD}\results:/home/jovyan/work/results" `
  ghcr.io/martc53/ansa-ml:latest
```

### Step 4B. Start Docker with a GPU

Use this command only if Docker can access an NVIDIA GPU. GPU access is recommended for model training and Optuna optimization.

Ubuntu/Linux:

```bash
docker run --gpus all -p 8888:8888 \
  -v /path/to/ANSA_OSF_data:/home/jovyan/work/data \
  -v "$(pwd)/models:/home/jovyan/work/models" \
  -v "$(pwd)/results:/home/jovyan/work/results" \
  ghcr.io/martc53/ansa-ml:latest
```

Windows PowerShell:

```powershell
docker run --gpus all -p 8888:8888 `
  -v "C:\path\to\ANSA_OSF_data:/home/jovyan/work/data" `
  -v "${PWD}\models:/home/jovyan/work/models" `
  -v "${PWD}\results:/home/jovyan/work/results" `
  ghcr.io/martc53/ansa-ml:latest
```

If GPU startup fails with an NVIDIA, WSL, or adapter error, rerun the CPU command above without `--gpus all`.

### Step 5. Open JupyterLab

After the container starts, open:

```text
http://localhost:8888
```

The notebooks are available inside the container at:

```text
/home/jovyan/work/notebooks/
```

The mounted OSF data are available at:

```text
/home/jovyan/work/data/
```

## Initialize OSF data inside Docker

After starting the container, open a JupyterLab terminal and run:

```bash
METADATA_DIR=/home/jovyan/work/data/Metadata \
CLINICAL_MANIFEST=/home/jovyan/work/data/Metadata/clinical_split_manifest.csv \
LOGARITHMIC_MANIFEST=/home/jovyan/work/data/Metadata/logarithmic_split_manifest.csv \
TENSOR_SOURCE_ROOT=/home/jovyan/work/data/processed_reduced_tensors \
CLINICAL_OUTPUT_ROOT=/home/jovyan/work/data/reduced \
LOGARITHMIC_OUTPUT_ROOT=/home/jovyan/work/data/reduced_log \
python scripts/rebuild_data_from_manifest.py
```

This command creates the organized clinical and logarithmic datasets from the unordered OSF tensor folder and the metadata manifests.

Expected outputs:

```text
/home/jovyan/work/data/reduced/
/home/jovyan/work/data/reduced_log/
```

Because `/home/jovyan/work/data` is mounted from your computer, the rebuilt folders will also appear in whatever local OSF folder you mounted, such as `ANSA_OSF_data/`.

## Notebook paths

The notebooks should use the following dataset paths inside Docker:

```python
from pathlib import Path

CLINICAL_DATA_ROOT = Path("/home/jovyan/work/data/reduced")
LOGARITHMIC_DATA_ROOT = Path("/home/jovyan/work/data/reduced_log")
MODEL_ROOT = Path("/home/jovyan/work/models")
RESULTS_ROOT = Path("/home/jovyan/work/results")
```

## Recommended notebook order

1. `notebooks/Optuna_clinical_model.ipynb`  
   Run hyperparameter optimization for the clinical classification task.

2. `notebooks/Clinical_model.ipynb`  
   Train and evaluate the final clinical model.

3. `notebooks/Optuna_logarithmic_model.ipynb`  
   Run hyperparameter optimization for the logarithmic classification task.

4. `notebooks/Logarithmic_model.ipynb`  
   Train and evaluate the final logarithmic model.

The Optuna notebooks are computationally expensive and are not required if you only want to rerun the final selected model configurations.

## Option 2: Run without Docker

A local Python environment can also be used. The original development environment used Python 3.9 with PyTorch, torchvision, scikit-learn, scikit-image, pandas, matplotlib, seaborn, and Optuna.

Create and activate an environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If running locally without Docker, update notebook paths to point to your local OSF data folder, model folder, and results folder. For example, if the OSF folder is located at `PROJECT_ROOT / "ANSA_OSF_data"`:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
OSF_DATA_ROOT = PROJECT_ROOT / "ANSA_OSF_data"
CLINICAL_DATA_ROOT = OSF_DATA_ROOT / "reduced"
LOGARITHMIC_DATA_ROOT = OSF_DATA_ROOT / "reduced_log"
MODEL_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
```

Then rebuild the OSF data locally:

```bash
METADATA_DIR=/path/to/ANSA_OSF_data/Metadata \
CLINICAL_MANIFEST=/path/to/ANSA_OSF_data/Metadata/clinical_split_manifest.csv \
LOGARITHMIC_MANIFEST=/path/to/ANSA_OSF_data/Metadata/logarithmic_split_manifest.csv \
TENSOR_SOURCE_ROOT=/path/to/ANSA_OSF_data/processed_reduced_tensors \
CLINICAL_OUTPUT_ROOT=/path/to/ANSA_OSF_data/reduced \
LOGARITHMIC_OUTPUT_ROOT=/path/to/ANSA_OSF_data/reduced_log \
python scripts/rebuild_data_from_manifest.py
```

On Windows PowerShell:

```powershell
$env:METADATA_DIR="C:\path\to\ANSA_OSF_data\Metadata"
$env:CLINICAL_MANIFEST="C:\path\to\ANSA_OSF_data\Metadata\clinical_split_manifest.csv"
$env:LOGARITHMIC_MANIFEST="C:\path\to\ANSA_OSF_data\Metadata\logarithmic_split_manifest.csv"
$env:TENSOR_SOURCE_ROOT="C:\path\to\ANSA_OSF_data\processed_reduced_tensors"
$env:CLINICAL_OUTPUT_ROOT="C:\path\to\ANSA_OSF_data\reduced"
$env:LOGARITHMIC_OUTPUT_ROOT="C:\path\to\ANSA_OSF_data\reduced_log"
python scripts/rebuild_data_from_manifest.py
```

## Expected model inputs

Each sample is a reduced multi-frame tensor derived from a time-resolved ANSA fluorescence image stack. The model uses seven image channels sampled from the reaction trajectory: a baseline frame generated by averaging early frames and six additional time points from the reaction. The first convolutional layer of ResNet is modified to accept seven input channels instead of the default three RGB channels.

## Outputs

Typical outputs include:

- trained model checkpoints in `models/`
- confusion matrices and performance figures in `results/`
- Optuna study outputs or best hyperparameter summaries, if generated by the optimization notebooks

## Notes for reproducibility

- The Docker image does not contain the OSF dataset.
- Always mount the local OSF data folder, wherever it lives on your computer, to `/home/jovyan/work/data` when running Docker.
- Rebuilding from manifests is deterministic with respect to the manifest files and tensor filenames.
- The script optionally verifies SHA256 checksums when the manifest contains a `sha256` column.
- GPU acceleration is recommended for model training and Optuna optimization, but the container can start and notebooks can run on CPU.

## Citation

A manuscript describing this workflow is in preparation for PLOS Computational Biology. A pre-print is available on [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.06.24.734232)

```text
Martin CD, Benson NC, Gummalla NS, Shimazu KN, Bender AT, Beck DAC, Posner JD.
ResNet Analysis for Semi-Quantitative Isothermal Molecular Testing.
Manuscript in preparation.
```

## License

Add license information here before public release.

## Contact

For questions about this repository, please contact the corresponding author listed in the manuscript or open an issue on GitHub.
