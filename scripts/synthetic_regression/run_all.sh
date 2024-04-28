#!/bin/zsh

export HYDRA_FULL_ERROR=1

python src/main.py --config-name train_val_lars_synthetic_regression
python src/main.py --config-name train_val_lasso_synthetic_regression
python src/main.py --config-name train_val_ridge_synthetic_regression
python src/main.py --config-name train_val_lasso_gd_synthetic_regression
python src/main.py --config-name train_val_ridge_gd_synthetic_regression