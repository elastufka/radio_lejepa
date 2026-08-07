#!/bin/sh
#SBATCH --job-name lejepa          # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/lejepa-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/lejepa-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 8             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=80G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
##SBATCH --begin="10:00"
##SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu002,gpu004,gpu006,gpu008,gpu009,gpu010,gpu046

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="LeJepa_RGZ"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PATH=/home/users/l/lastufka/.conda/envs/dinov3/bin:$PATH
alias mysrun='srun /home/users/l/lastufka/.conda/envs/dinov3/bin/python3'

#/home/users/l/lastufka/scratch/Galaxy10DECals
#/home/users/l/lastufka/scratch/MGCLS_data/enhanced/crops_224_3chan_rescale
BACKBONE="EUPE-ViT-S"
LR=1.5e-4
RUN_NAME="${BACKBONE}_MGCLSonly${SIZE}_lr${LR}"
OUT_DIR="/home/users/l/lastufka/scratch/lejepa_rgz/"

#if [ ! -d "$OUT_DIR" ]; then
    mysrun -m accelerate.commands.launch --num_processes 2 /home/users/l/lastufka/eupe_paper/lejepa_rgz.py 
        # --model vit_small \
        # --lr $LR \
        # --output_dir $OUT_DIR \
        # --batch_size 64 \
        # --epochs 10
#fi