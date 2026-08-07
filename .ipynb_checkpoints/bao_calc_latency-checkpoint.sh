#!/bin/sh
#SBATCH --job-name latency          # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/latency-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/latency-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 8             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=32G
##SBATCH --gres=gpu:a100:1
#SBATCH --time 01:00:00                  # maximum run time.
##SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu002,gpu004,gpu006,gpu008,gpu009,gpu010,gpu046

#export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
#export WANDB_PROJECT="supervised_RGZ"

#unset PYTHONPATH
#export PYTHONNOUSERSITE=1
export PATH=/home/users/l/lastufka/.conda/envs/dinov3/bin:$PATH
alias mysrun='srun /home/users/l/lastufka/.conda/envs/dinov3/bin/python3'

mysrun /home/users/l/lastufka/eupe_paper/calc_latency.py \
        --model vit_huge_plus_patch16_dinov3.lvd1689m # \
        #--dataset2 /home/users/l/lastufka/scratch/GalaxyMNIST --umap #MiraBest #/home/users/l/lastufka/scratch/FIRSTGalaxyData/pngs #/home/users/l/lastufka/scratch/rgz/od /home/users/l/lastufka/scratch/byol_rgz20k/byol_model.pth /home/users/l/lastufka/scratch/lejepa/efficientnet_rgz20k/checkpoint.pt