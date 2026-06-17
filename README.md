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
├── src/
│   └── ansa/
│       ├── data_utils.py
│       └── visualization.py
├── data/              # created locally; not tracked by git
├── models/            # created locally; not tracked by git
└── results/           # created locally; not tracked by git
```

The `data/`, `models/`, and `results/` directories are intentionally kept out of version control. The processed tensor data should be downloaded separately from OSF.

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

After rebuilding, the local data folder should look like this:

```text
data/
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

- `data/reduced/` is used by the clinical model.
- `data/reduced_log/` is used by the logarithmic model.

## Option 1: Run with Docker

Docker is the recommended way to reproduce the analysis environment. The Docker image contains the software environment, notebooks, source code, and rebuild script. The image does **not** contain the ANSA dataset. Download the dataset separately from OSF and mount it into the container at runtime.

### Step 1. Build the image locally

From the repository root:

```bash
docker build -t ansa-ml .
```

### Step 2. Prepare local folders

Create local folders for mounted data, trained models, and output figures. These folders are not tracked by git.

Ubuntu/Linux:

```bash
mkdir -p data models results
```

Windows PowerShell:

```powershell
mkdir data
mkdir models
mkdir results
```

### Step 3. Download and place the OSF data

Download the OSF dataset separately and place the extracted contents inside the local `data/` folder. Before starting Docker, the repository should look like this:

```text
ANSA/
├── Dockerfile
├── notebooks/
├── scripts/
├── src/
├── data/
│   ├── Metadata/
│   │   ├── clinical_split_manifest.csv
│   │   ├── logarithmic_split_manifest.csv
│   │   └── sample_manifest.csv
│   └── processed_reduced_tensors/
│       ├── *.pt
│       └── ...
├── models/
└── results/
```

The Docker run commands below mount the local `data/`, `models/`, and `results/` folders into the container. The most important mount is:

```text
local data folder -> /home/jovyan/work/data inside Docker
```

The notebooks and rebuild script assume that the OSF data are available inside the container at:

```text
/home/jovyan/work/data
```

### Step 4A. Start Docker without a GPU

Use this command if Docker cannot access an NVIDIA GPU, or if you want to run CPU-only. This is sufficient for opening the notebooks, rebuilding the dataset, and testing the workflow. Training and Optuna optimization will be slower on CPU.

Ubuntu/Linux:

```bash
docker run -p 8888:8888 \
  -v "$(pwd)/data:/home/jovyan/work/data" \
  -v "$(pwd)/models:/home/jovyan/work/models" \
  -v "$(pwd)/results:/home/jovyan/work/results" \
  ansa-ml
```

Windows PowerShell:

```powershell
docker run -p 8888:8888 `
  -v "${PWD}\data:/home/jovyan/work/data" `
  -v "${PWD}\models:/home/jovyan/work/models" `
  -v "${PWD}\results:/home/jovyan/work/results" `
  ansa-ml
```

### Step 4B. Start Docker with a GPU

Use this command only if Docker can access an NVIDIA GPU. GPU access is recommended for model training and Optuna optimization.

Ubuntu/Linux:

```bash
docker run --gpus all -p 8888:8888 \
  -v "$(pwd)/data:/home/jovyan/work/data" \
  -v "$(pwd)/models:/home/jovyan/work/models" \
  -v "$(pwd)/results:/home/jovyan/work/results" \
  ansa-ml
```

Windows PowerShell:

```powershell
docker run --gpus all -p 8888:8888 `
  -v "${PWD}\data:/home/jovyan/work/data" `
  -v "${PWD}\models:/home/jovyan/work/models" `
  -v "${PWD}\results:/home/jovyan/work/results" `
  ansa-ml
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

Because `/home/jovyan/work/data` is mounted from your computer, the rebuilt folders will also appear in your local `data/` directory.

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

If running locally, update notebook paths to point to your local `data/`, `models/`, and `results/` folders, for example:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
CLINICAL_DATA_ROOT = PROJECT_ROOT / "data" / "reduced"
LOGARITHMIC_DATA_ROOT = PROJECT_ROOT / "data" / "reduced_log"
MODEL_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
```

Then rebuild the OSF data locally:

```bash
METADATA_DIR=data/Metadata \
CLINICAL_MANIFEST=data/Metadata/clinical_split_manifest.csv \
LOGARITHMIC_MANIFEST=data/Metadata/logarithmic_split_manifest.csv \
TENSOR_SOURCE_ROOT=data/processed_reduced_tensors \
CLINICAL_OUTPUT_ROOT=data/reduced \
LOGARITHMIC_OUTPUT_ROOT=data/reduced_log \
python scripts/rebuild_data_from_manifest.py
```

On Windows PowerShell:

```powershell
$env:METADATA_DIR="data/Metadata"
$env:CLINICAL_MANIFEST="data/Metadata/clinical_split_manifest.csv"
$env:LOGARITHMIC_MANIFEST="data/Metadata/logarithmic_split_manifest.csv"
$env:TENSOR_SOURCE_ROOT="data/processed_reduced_tensors"
$env:CLINICAL_OUTPUT_ROOT="data/reduced"
$env:LOGARITHMIC_OUTPUT_ROOT="data/reduced_log"
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
- Always mount the OSF data folder to `/home/jovyan/work/data` when running Docker.
- Rebuilding from manifests is deterministic with respect to the manifest files and tensor filenames.
- The script optionally verifies SHA256 checksums when the manifest contains a `sha256` column.
- GPU acceleration is recommended for model training and Optuna optimization, but the container can start and notebooks can run on CPU.

## Citation

A manuscript describing this workflow is in preparation for PLOS Computational Biology. Citation details will be added after publication.

```text
Martin CD, Benson NC, Gummalla NS, Shimazu KN, Bender AT, Beck DAC, Posner JD.
ResNet Analysis for Semi-Quantitative Isothermal Molecular Testing.
Manuscript in preparation.
```

## License

Add license information here before public release.

## Contact

For questions about this repository, please contact the corresponding author listed in the manuscript or open an issue on GitHub.
