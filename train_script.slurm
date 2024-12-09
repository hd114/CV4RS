#!/bin/bash
#SBATCH --job-name=pytorch_job
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/paul/projects/CV4RS-orig
#SBATCH --export=ALL,CONFIGS_PATH=CV4RS-orig/configs/test-config-resnet-p.yaml
#SBATCH --output=log/pytorch_job_%j.out
#SBATCH --error=log/pytorch_job_%j.err

# Run your Python script
python ~/projects/CV4RS-orig/train.py
