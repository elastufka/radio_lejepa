#!/bin/sh
#SBATCH --job-name mgcls_yolo            # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/mgcls_yolo-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/mgcls_yolo-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 8             # number of cpus for each task. One by default
#SBATCH --partition shared-cpu         # the partition to use. By default debug-cpu
##SBATCH --gpus 1
#SBATCH --mem=50G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
##SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu002,gpu004,gpu006,gpu008,gpu009,gpu010,gpu046

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PATH=/home/users/l/lastufka/.conda/envs/dinov3/bin:$PATH
alias mysrun='srun /home/users/l/lastufka/.conda/envs/dinov3/bin/python3'

 mysrun /home/users/l/lastufka/eupe_paper/leyolo_detect.py  \
    --output_dir /home/users/l/lastufka/scratch/yolo/RGZ20k_ep29_512f_1e-4 \
    --epochs 100 --freeze_mode "static" \
    --imgsz 512 --lr0 1e-4 --jepa_ckpt /home/users/l/lastufka/scratch/lejepa/yolo8n_rgz20k/checkpoint.pt 

#--freeze_mode "static" --jepa_ckpt /home/users/l/lastufka/scratch/lejepa/yolo8n_rgz/checkpoint.pt \