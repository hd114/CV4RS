#!/bin/bash
#SBATCH --job-name=unp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
##SBATCH --time=02:00:00
##SBATCH --output=logs/t3sce2_randomunprotected_pr095_%j.out
##SBATCH --output=logs/sce2_random_pr05_%j.out
##SBATCH --output=logs/t3sce1_nisp_pr07_%j.out
##SBATCH --output=logs/t3sce1_unpruned_%j.out
##SBATCH --output=logs/t3sce2_nisp_pr07_%j.out
##SBATCH --output=logs/r18_sce2_epr4_pr2_unprotected_nisp_pr095_%j.out
##SBATCH --output=logs/r18_sce2_epr4_unpruned_%j.out
#SBATCH --error=logs/err/job_%j.err

# Run your Python script
python ~/projects/CV4RS-orig/train.py --scenario 2 --p_strategy nisp --p_rate 0.95 --p_round 4 --comm 40 --epr 1