# Experiments Folder Guide

All experiment entry scripts are centralized in `main/experiments`.

## File Roles

- `__init__.py`  
  Marks this directory as a Python package.

- `README.md`  
  This guide file; records script purposes and running conventions.

- `experiment_logger.py`  
  Generic command wrapper that writes full stdout/stderr to timestamped log files named by `date + model + params`.

- `compare_iosc_vs_fuzzy_effective.py`  
  Quick-run training/evaluation helper with short-epoch patches and unified candidate routing evaluator.  
  Also provides `QuickTrainerBase`, `QuickTrainerFuzzy`, and utility patch functions reused by other scripts.

- `item_importance_utils.py`  
  Item salience score builder. Supports:
  - `uniform`
  - `popularity`
  - `pagerank`
  - `eigenvector`
  - `blend:popularity:...,pagerank:...,eigenvector:...`

- `multi_dataset_benchmark.py`  
  Multi-model benchmark orchestrator on multiple datasets.  
  Exposes shared helpers such as:
  - `prepare_cfg`
  - `maybe_attach_item_importance`
  - `evaluate_with_builtin_eval_ranking`
  - `train_eval`

- `innovation_preserving_search.py`  
  Grid search focused on innovation-preserving Fuzzy configs (orthogonal decoupling + multi-factor salience) against ComiRec baseline.

- `four_dataset_unified_summary.py`  
  Unified-budget summary runner that reports Fuzzy vs ComiRec on `yelp/electronics/ml1m/cd`.

- `comirec_main.py`  
  Standalone ComiRec training entry with existing Trainer/data flow.

- `mind_main.py`  
  Standalone MIND training entry with existing Trainer/data flow.

## Run Convention

Run scripts from the `main` directory:

`py -3.12 experiments/experiment_logger.py --model Fuzzy --params orth0.01_rec0.5_conf0.3 -- py -3.12 experiments/innovation_preserving_search.py`
