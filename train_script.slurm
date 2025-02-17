#!/bin/bash
#SBATCH --job-name=unp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
##SBATCH --time=02:00:00
##SBATCH --output=logs/sce2_random_pr05_%j.out
##SBATCH --output=logs/t3sce1_unpruned_%j.out
#SBATCH --output=scrap/t3sce1_unpruned_%j.out
#SBATCH --error=logs/err/job_%j.err

# Run your Python script
python ./train.py --scenario 2 --p_strategy nisp --p_rate 0.95 --p_round 4 --comm 40 --epr 1