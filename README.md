# Bangla Clickbait Detection with Transformer Models

A comparative NLP project for classifying Bangla news headlines as **clickbait** (`1`) or **non-clickbait** (`0`). The experiments compare Bangla and multilingual pretrained language models against TF-IDF baselines, with a validation-selected transformer ensemble and additional error-analysis and interpretability experiments.

> **Status:** Research/portfolio project. Metrics below are the results saved in the current notebook run; they have not been independently reproduced from a clean environment.

## Results at a glance

The current notebook reports the following held-out test-set results on 2,988 examples:

| System | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| Top-3 transformer ensemble | **0.8286** | **0.8133** | **0.9033** |
| IndicBERTv2 | 0.8230 | 0.8093 | 0.8910 |
| XLM-RoBERTa-large | 0.8220 | 0.8085 | 0.8911 |
| TF-IDF + Logistic Regression | 0.7550 | 0.7356 | — |
| TF-IDF + Linear SVM | 0.7523 | 0.7297 | — |

Macro-F1 is emphasized because the dataset is imbalanced. The current cleaned dataset contained 10,469 training rows and 2,988 test rows. A stratified 90/10 split of the training data produced 9,422 training and 1,047 validation examples.

**Interpretation:** These results are a benchmark snapshot from the saved run, not a claim of state-of-the-art performance. Latency and memory measurements depend on the hardware and software environment.

## What this project covers

- Bangla headline preprocessing and exploratory data analysis.
- Stratified train/validation splitting and early stopping for encoder fine-tuning.
- Comparison of pretrained models, including XLM-RoBERTa, BanglaBERT, Bangla-BERT, mDeBERTa, MuRIL, IndicBERTv2, and TigerLLM-1B-it with QLoRA.
- TF-IDF + Logistic Regression and TF-IDF + Linear SVM baselines.
- A top-3 transformer logit-averaging ensemble.
- Evaluation with Accuracy, Macro-F1, class-level Precision/Recall/F1, ROC-AUC, and PR-AUC.
- Error diagnostics, feature-level ablations, calibration analysis, LIME/SHAP-related interpretation, bootstrap confidence intervals, and statistical comparisons.
- Export of metrics, plots, LaTeX tables, and a reusable model checkpoint from the notebook.

## Dataset

The project uses the [BanglaBait: Bangla Clickbait Dataset on Kaggle](https://www.kaggle.com/datasets/motaharmahtab/banglabait-bangla-clickbait-dataset). Only the `title` field is used as input and `label` as the target.

The dataset files are **not included** in this repository. Download them from the source and follow the dataset's terms and license. The licenses of the dataset and pretrained models may differ from the license chosen for this project's code.

## How to run

The current notebook is configured for **Kaggle**, with a GPU accelerator (the original run used two NVIDIA T4 GPUs).

1. Open `notebooks/banglabait_clickbait_detection.ipynb` in Kaggle (or import this notebook into a Kaggle notebook).
2. Use **Add Input** to attach the BanglaBait dataset so that `train.csv` and `test.csv` are available under `/kaggle/input/`.
3. Enable Internet access so model checkpoints and optional fonts/dependencies can be downloaded.
4. Select a GPU accelerator. Two T4 GPUs match the notebook's documented configuration; a single GPU may require smaller batch sizes, and CPU execution will be much slower.
5. Run the notebook from the first cell. It installs the additional libraries it needs and writes generated artifacts to `/kaggle/working/banglabait_results/`.

The notebook deliberately avoids reinstalling PyTorch in the Kaggle environment to reduce the risk of breaking the preconfigured CUDA/PyTorch installation. For local execution, adapt the data paths and install a PyTorch build compatible with the machine's CUDA/CPU environment before running the experiments.

## Recommended repository layout

```text
bangla-clickbait-detection/
├── README.md
├── LICENSE
├── .gitignore
├── requirements-kaggle.txt
├── notebooks/
│   └── banglabait_clickbait_detection.ipynb
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── confusion_matrix_ensemble.png
│   └── error_analysis.md
└── docs/
    └── methodology.md
```

Add the files under `results/` only after exporting the corresponding outputs from the notebook. Keep this repository focused on small, useful artifacts rather than committing every intermediate file or model checkpoint. If the project grows beyond a research notebook, move reusable preprocessing, training, and evaluation functions into a `src/` package and add automated tests under `tests/`.

## Reproducibility and evaluation notes

- Fix random seeds and record the dataset source, package versions, hardware, and experiment configuration alongside each run.
- Use the validation split for model/checkpoint selection and reserve the test set for final reporting. In the current notebook, the single-encoder export cell selects from a summary ranked by test Macro-F1; change that selection to use validation results before presenting the exported single-encoder checkpoint as the final selected model.
- Check for duplicate titles across the train and test files before the final run, and document any overlap-handling policy.
- Report whether each result is a saved historical run or was reproduced from a clean run.

## Limitations

- Results are measured on one dataset and should not be assumed to generalize to every Bangla news source, time period, or style of headline.
- The class distribution is imbalanced, so accuracy alone does not describe model performance.
- Interpretability methods and heuristic error categories provide diagnostic evidence; they do not by themselves prove that a model's decision process is causally faithful.
- Inference latency depends on the accelerator, batch size, sequence length, and serving setup.

## Acknowledgments

Credit the BanglaBait dataset authors and the maintainers of all pretrained models used in this work. Follow each source's citation and usage requirements.
