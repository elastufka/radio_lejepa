#!/bin/sh
#SBATCH --job-name lejepa_probe          # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/lejepa_probe-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/lejepa_probe-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 8             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 2
#SBATCH --mem=80G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
##SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu002,gpu004,gpu006,gpu008,gpu009,gpu010,gpu046

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="supervised_RGZ"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PATH=/home/users/l/lastufka/.conda/envs/dinov3/bin:$PATH
alias mysrun='srun /home/users/l/lastufka/.conda/envs/dinov3/bin/python3'

#/home/users/l/lastufka/scratch/Galaxy10DECals
#/home/users/l/lastufka/scratch/GalaxyMNIST
#/home/users/l/lastufka/scratch/MGCLS_data/enhanced/crops_224_3chan_rescale
BACKBONE="vitt16_gz10224"
LR=1e-3
RUN_NAME="${BACKBONE}_gmnist_lp_lr${LR}"
OUT_DIR="/home/users/l/lastufka/scratch/FM_compare/RGZ/${RUN_NAME}"

SEEDS=(14 27 88)
for SEED in "${SEEDS[@]}"; do
#if [ ! -d "$OUT_DIR" ]; then
    mysrun -m accelerate.commands.launch --num_processes 2 /home/users/l/lastufka/eupe_paper/lejepa_probe.py \
        --model_name "/home/users/l/lastufka/scratch/lejepa/${BACKBONE}/checkpoint_99.pt" \
        --output_dir $OUT_DIR \
        --dataset_train "/home/users/l/lastufka/scratch/GalaxyMNIST" \
        --train_batch_size 64 \
        --run_name $RUN_NAME \
        --lr $LR \
        --epochs 100 --freeze_backbone --cv --return_dict --seed $SEED
#fi
done