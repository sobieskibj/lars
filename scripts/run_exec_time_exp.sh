#!/usr/bin/env bash
#SBATCH --partition short
#SBATCH --account=mi2lab-hi
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 24:00:00
#SBATCH --job-name=inp_exp
#SBATCH --output=slurm_logs/%A.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

# load slurm
source /etc/profile.d/slurm.sh

# activate env
source /raid/shared/$USER/conda/etc/profile.d/conda.sh
conda activate inp_exp

# run exp
cd /home2/faculty/bsobieski/lars

export HYDRA_FULL_ERROR=1
wandb online

N=10_000
PS=($(seq -s ' ' 100 100 10000))

for P in ${PS[@]}; do
    
    python src/main.py --config-name train_val_lars_synthetic_regression exp.n=$N exp.p=$P
    
    python src/main.py --config-name train_val_lars_sk_synthetic_regression exp.n=$N exp.p=$P

done