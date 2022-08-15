#!/bin/bash
#SBATCH --job-name=HAE
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

eval "$(/opt/app/anaconda3/bin/conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib

conda create -n HAE python=3.7 pytorch=1.10 cudatoolkit=11.1 torchvision -c pytorch -y
conda activate HAE
pip3 install openmim
mim install mmcv-full
pip3 install -e .
