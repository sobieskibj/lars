#!/bin/zsh

export HYDRA_FULL_ERROR=1

python src/main.py --config-name train_val_lars_diabetes
python src/main.py --config-name train_val_lasso_diabetes
python src/main.py --config-name train_val_ridge_diabetes
python src/main.py --config-name train_val_lasso_gd_diabetes
python src/main.py --config-name train_val_ridge_gd_diabetes