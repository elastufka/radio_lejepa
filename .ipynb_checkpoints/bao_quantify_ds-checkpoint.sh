#!/bin/sh
#SBATCH --job-name quantify          # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/quantify-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/quantify-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 8             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=40G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
##SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu002,gpu004,gpu006,gpu008,gpu009,gpu010,gpu046

#export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
#export WANDB_PROJECT="supervised_RGZ"

#unset PYTHONPATH
#export PYTHONNOUSERSITE=1
export PATH=/home/users/l/lastufka/.conda/envs/dinov3/bin:$PATH
alias mysrun='srun /home/users/l/lastufka/.conda/envs/dinov3/bin/python3'

mysrun /home/users/l/lastufka/eupe_paper/embed_metrics.py \
        --model_name /home/users/l/lastufka/scratch/lejepa/vitt16_rgz20k_224_vl8_MB_lr5e-4_81/checkpoint-327/pytorch_model.bin --umap \
        #--dataset_train /home/users/l/lastufka/eupe_paper/eupet_new_embeddings.pkl # \
        #--dataset2 /home/users/l/lastufka/scratch/GalaxyMNIST --umap #MiraBest #/home/users/l/lastufka/scratch/FIRSTGalaxyData/pngs #/home/users/l/lastufka/scratch/rgz/od /home/users/l/lastufka/scratch/byol_rgz20k/byol_model.pth /home/users/l/lastufka/scratch/lejepa/efficientnet_rgz20k/checkpoint.pt