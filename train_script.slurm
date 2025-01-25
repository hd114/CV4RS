#!/bin/bash
#SBATCH --job-name=finalpres
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
##SBATCH --time=02:00:00
##SBATCH --output=logs/sce1_random_pr03_job_%j.out
#SBATCH --output=logs/sce2_l2_pr03_job_%j.out
#SBATCH --error=logs/err/job_%j.err

# Run your Python script
python ~/projects/CV4RS-orig/train.py --scenario 2 --p_strategy l2 --p_rate 0.3 --p_round 4 --comm 40 --epr 1
