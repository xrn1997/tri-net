#!/bin/bash
#SBATCH -J tri-net
#SBATCH -o result.txt								
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=572249217@qq.com
#SBATCH -p dlq
python main.py