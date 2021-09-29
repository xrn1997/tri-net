#!/bin/bash
#SBATCH -J tri-net
#SBATCH -o result.txt								
#SBATCH -N 1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:3
#SBATCH --mem=110G
#SBATCH --mail-type=END
#SBATCH --mail-user=572249217@qq.com
python main.py

