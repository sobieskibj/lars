#!/bin/zsh

export HYDRA_FULL_ERROR=1

## Experiment 1 ##

N=100
P=10

python src/main.py --config-name train_val_lars_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_lars_sk_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_lasso_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_ridge_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_lasso_gd_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_ridge_gd_synthetic_regression exp.n=$N exp.p=$P

## Experiment 2 ##

N=100
P=200

python src/main.py --config-name train_val_lars_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_lars_sk_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_lasso_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_ridge_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_lasso_gd_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_ridge_gd_synthetic_regression exp.n=$N exp.p=$P

## Experiment 3 ##

N=10000
P=100

python src/main.py --config-name train_val_lars_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_lars_sk_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_lasso_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_ridge_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_lasso_gd_synthetic_regression exp.n=$N exp.p=$P
python src/main.py --config-name train_val_ridge_gd_synthetic_regression exp.n=$N exp.p=$P

## Experiment 4 ##

python src/main.py --config-name train_val_lars_diabetes
python src/main.py --config-name train_val_lars_sk_diabetes
python src/main.py --config-name train_val_lasso_diabetes
python src/main.py --config-name train_val_ridge_diabetes
python src/main.py --config-name train_val_lasso_gd_diabetes 
python src/main.py --config-name train_val_ridge_gd_diabetes