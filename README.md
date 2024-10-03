# A Bayesian analysis of heart rate variability changes over acute episodes of bipolar disorder

This codebase was developed by [Filippo Corponi](https://github.com/FilippoCMC) and [Bryan M. Li](https://github.com/bryanlimy). It is part of the npj Mental Health paper "[A Bayesian analysis of heart rate variability changes over acute episodes of bipolar disorder](https://www.nature.com/articles/s44184-024-00090-x)". If you find this code or any of the ideas in the paper useful, please consider starring this repository and citing:

```bibtex
@article{corponi2024bayesian,
  title = {A Bayesian analysis of heart rate variability changes over acute episodes of bipolar disorder},
  author = {Corponi, F. and Li, B.M. and Anmella, G. \etal},
  journal = {npj Mental Health Research},
  volume = {3},
  page = {44},
  year = {2024},
  doi = {10.1038/s44184-024-00090-x},
  url = {https://doi.org/10.1038/s44184-024-00090-x}
}
```

## Setup
- Create a new [conda](https://docs.anaconda.com/miniconda/) environment with Python 3.10.
  ```bash
  conda create -n hrv python=3.10
  ```
- Activate `hrv` virtual environment
  ```bash
  conda activate hrv
  ```
- Install all dependencies and packages.
  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```
[dataset/README.md](dataset/README.md) details the structure of the dataset.

## Preprocessing

The commands below preprocess the data (see manuscript for details). Please see `--help` for all available options.

  ```bash
  python preprocess_ds.py --output_dir data/preprocessed/unsegmented --overwrite --overwrite_spreadsheet
  ```

  ```bash
  python segment.py --output_dir data/preprocessed/sl300_ss60 --extract_features hrv --hrv_extractor flirt --segment_length 300 --step_size 60 --overwrite --use_empatica_ibi
  ```

  ```bash
  python build_hrv_dataset.py --dataset data/preprocessed/sl300_ss60 --output_dir runs
  ```

## Bayesian Analysis

The command below runs the HRV analysis (see manuscript for details). Please see `--help` for all available options.
  ```bash
  python bayesian_analysis.py --working_dir runs
  ```
