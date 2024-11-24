#!/bin/bash
#SBATCH --job-name=pytorch_debug_job
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=02:00:00
#SBATCH --output=log/pytorch_debug_%j.out
#SBATCH --error=log/pytorch_debug_%j.err

# Aktiviert dein Python-Environment
#source ~/micromamba/bin/activate pytorch_env

# Debugpy startet und wartet auf eine Verbindung
python -m debugpy --listen 0.0.0.0:9680 --wait-for-client train.py
