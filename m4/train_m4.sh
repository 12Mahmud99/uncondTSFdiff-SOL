#!/usr/bin/env bash
#SBATCH --job-name=uncondTSFdiff
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=uber.%j.out
#SBATCH --error=uber.%j.err

module purge
module load cuda-12.8.1-gcc-12.1.0

ls $CUDA_PATH/include/cuda.h
ls $CUDA_PATH/include/nvrtc.h

rm -rf ~/.cache/keops*

python bin/train_model.py -c configs/train_tsdiff/train_m4.yaml
