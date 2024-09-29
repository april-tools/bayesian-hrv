# A Bayesian analysis of heart rate variability changes over acute episodes of bipolar disorder

This codebase was developed by [Filippo Corponi](https://github.com/FilippoCMC) and [Bryan M. Li](https://github.com/bryanlimy). It is part of the npj Mental Health paper "[A Bayesian analysis of heart rate variability changes over acute episodes of bipolar disorder](https://osf.io/preprints/psyarxiv)". If you find this code or any of the ideas in the paper useful, please consider starring this repository and citing:

```buildoutcfg
@article{
author = {},
title = {},
year = {},
isbn = {},
publisher = {},
address = {},
url = {},
doi = {},
pages = {},
numpages = {11},
keywords = {},
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
