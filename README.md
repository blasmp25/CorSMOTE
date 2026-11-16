# CorSMOTE: A Correlation-Aware Oversampling Technique for Class-Imbalanced Learning

CorSMOTE is an oversampling method for imbalanced classification. Classical SMOTE-based
methods treat every feature as equally relevant when generating synthetic samples. CorSMOTE
instead uses **Mutual Information**, computed over quantile-discretized attributes, to identify
the features most related to the class label, and **only modifies those features** when
generating new minority instances. The remaining attributes keep the values of the original
minority instance.

This repository contains the implementation of CorSMOTE and the code used to run the
experiments reported in the paper.

> **Paper:** F.-B. Martínez-Pareja, J.-F. Cabrera-Sánchez, D. Dorvigny Dorvigny, E.-L. Silva-Ramírez.
> *CorSMOTE: A Correlation-Aware Oversampling Technique for Class-Imbalanced Learning.*
> Submitted to *Expert Systems with Applications*.

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/<user>/CorSMOTE.git
cd CorSMOTE
pip install -r requirements.txt
```

## Using CorSMOTE

`CorSMOTE` follows the `imbalanced-learn` interface, so it can be used like any other sampler,
including inside an `imblearn.pipeline.Pipeline` and `GridSearchCV`.

```python
import sys
sys.path.append("src")
from cor_smote import CorSMOTE

sampler = CorSMOTE(k_neighbors=5, top_k_features=5)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
```

Main parameters:

| Parameter | Description |
|---|---|
| `k_neighbors` | Number of nearest neighbours within the minority class. |
| `top_k_features` | Number of most relevant features (by Mutual Information) modified during generation. |
| `sampling_strategy` | `'auto'` to balance every minority class with the majority class, or a dict `{class: n_samples}`. |

Features are expected to be numerical and standardized (see `src/data_setup.py`).

## Reproducing the experiments

### 1. Datasets

The datasets are **not included** in this repository. See [`Datasets/README.md`](Datasets/README.md)
for their sources and the expected format, and place the CSV files in the `Datasets/` folder.

### 2. Run the experimental pipeline

Run the scripts from the repository root:

```bash
# All datasets, samplers, classifiers and 30 seeds (as in the paper)
python src/pipeline.py

# A subset, e.g. one dataset, two samplers, one classifier and one seed
python src/pipeline.py --datasets glass4.csv --samplers smote corsmote --models rf --seed 0
```

Available options:

- `--samplers`: `none`, `smote`, `borderline`, `adasyn`, `corsmote`, `safe_level_smote`, `smote_ipf`, `mdo`
- `--models`: `dt`, `rf`, `bagging`, `lightgbm`, `xgboost`, `mlp`, `tabnet`, `svm`
- `--seed`: run a single seed (default: seeds 0 to 29)

Safe-Level-SMOTE, SMOTE-IPF and MDO are taken from the
[`smote-variants`](https://github.com/analyticalmindsltd/smote_variants) library.

### 3. Computational cost

```bash
python src/time_optimization.py --dataset glass4.csv --seed 0
```

### 4. Analysis and figures

The statistical analysis (Friedman, Nemenyi, Scott-Knott ESD) and the figures of the paper are
produced in `src/Analysis.ipynb` and `src/analyze_results.py`.

## Repository structure

```
CorSMOTE/
├── Datasets/                  # place the dataset CSV files here (not included)
└── src/
    ├── cor_smote.py           # CorSMOTE implementation
    ├── additional_samplers.py # wrapper for smote-variants samplers
    ├── data_setup.py          # loading, train/test split and scaling
    ├── experiments.py         # hyperparameter tuning and evaluation
    ├── pipeline.py            # main experimental pipeline
    ├── time_optimization.py   # computational cost measurement
    ├── analyze_results.py     # results aggregation
    ├── utils.py               # helper functions
    └── Analysis.ipynb         # statistical analysis and figures
```

## License

This project is released under the [MIT License](LICENSE).
