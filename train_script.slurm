#!/bin/bash
#SBATCH --job-name=finalpres
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
##SBATCH --time=02:00:00
##SBATCH --output=logs/sce1_nisp_pr05_job_%j.out
#SBATCH --output=logs/sce1_nisp_pr09_job_%j.out
#SBATCH --error=logs/err/job_%j.err

# Run your Python script
python ~/projects/CV4RS-orig/train.py --scenario 1 --p_strategy nisp --p_rate 0.9 --p_round 4 --comm 40 --epr 1
