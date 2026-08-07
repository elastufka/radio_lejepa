#!/bin/sh
#SBATCH --job-name eval_lejepa           # this is a parameter to help you sort your job when listing it
#SBATCH --error /home/users/l/lastufka/sbatch_logs/eval_lejepa-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output /home/users/l/lastufka/sbatch_logs/eval_lejepa-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 8             # number of cpus for each task. One by default
#SBATCH --partition shared-gpu         # the partition to use. By default debug-cpu
#SBATCH --gpus 1
#SBATCH --mem=50G
##SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time 12:00:00                  # maximum run time.
##SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu002,gpu004,gpu006,gpu008,gpu009,gpu010,gpu046

export WANDB_API_KEY=58fb50375582e0745d756777d78cd214b3b39e92
export WANDB_PROJECT="supervised_RGZ"

#unset PYTHONPATH
#export PYTHONNOUSERSITE=1
export PATH=/home/users/l/lastufka/.conda/envs/dinov3/bin:$PATH
alias mysrun='srun /home/users/l/lastufka/.conda/envs/dinov3/bin/python3'

SIZES=(30 50) # 30 50)
SEEDS=(86 56 47)
LR=1e-6 #5e-4
DS=("MB") # "FIRST") #"FIRST" "MB")
STATE="finetune"
#probe
for SEED in "${SEEDS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
#"/home/users/l/lastufka/scratch/FM_compare/MGCLS_distill/$BACKBONE/checkpoint-20800"
#"/home/users/l/lastufka/scratch/FM_compare/RGZ_distill/$BACKBONE/checkpoint-12500"
#"/home/users/l/lastufka/scratch/lejepa/${BACKBONE}/checkpoint_99.pt"
#"/home/users/l/lastufka/scratch/lejepa/${BACKBONE}/checkpoint_40.pt"  -m accelerate.commands.launch --num_processes 2 
#/home/users/l/lastufka/scratch/"${BACKBONE}"/simclr_vit.pt
#/home/users/l/lastufka/scratch/"${BACKBONE}"/byol_model.pth
#/home/users/l/lastufka/scratch/lejepa/"${BACKBONE}"/checkpoint.pt
        BACKBONE="EUPE-ViT-T" #"efficientnet_rgz20k" #"vitt16_rgz20k_224_vl8" #"efficientnet_rgz20k" #"vitt16_rgz20k_224_vl8" #"EUPE-ConvNeXT-T" #"efficientnet_rgz" #T_RGZ_lr1e-5" #"byol_rgz" #"EUPE-ViT-T_MGCLSonly_lr1e-5" #"vitt16_rgz20k_224_vl8" #FIRSTGalaxyData/pngs
    for D in "${DS[@]}"; do
        #for LR in "${LRS[@]}"; do
            if [[ "$D" == "FIRST" ]]; then
                EXTRA_ARGS=(--dataset_train "/home/users/l/lastufka/scratch/FIRSTGalaxyData/pngs" --flip --jitter --center_crop 200 --normalize) #--center_crop 200
            elif [[ "$D" == "RGZ" ]]; then
                EXTRA_ARGS=(--dataset_train "/home/users/l/lastufka/scratch/rgz" --fake_chan --flip --jitter --normalize)
            else
                EXTRA_ARGS=(--dataset_train "/home/users/l/lastufka/scratch/MiraBest" --fake_chan --flip --jitter --center_crop 150)
            fi
            
            if [[ "$STATE" == "frozen" ]]; then
                EXTRA_ARGS+=(--freeze_backbone 1)
                RUN_NAME="${BACKBONE}_"${D}"_lr${LR}_${SIZE}_${SEED}f" 
            else
                EXTRA_ARGS+=(--both)
                RUN_NAME="${BACKBONE}_"${D}"_lr${LR}_${SIZE}_${SEED}"
            fi
            OUT_DIR="/home/users/l/lastufka/scratch/FM_compare/${D}/${RUN_NAME}"
            #if [ ! -d "$OUT_DIR" ]; then
                mysrun /home/users/l/lastufka/FM_compare/finetune.py \
                    --model_name $BACKBONE  \
                    --lr $LR \
                    --output_dir $OUT_DIR \
                    --run_name $RUN_NAME \
                    --patience 200 --epochs 300 --eid 1 --num_workers 8 --weight_decay 0.05 \
                     --ims_per_batch 64 --size 224 --seed $SEED --nlabels $SIZE  "${EXTRA_ARGS[@]}" 
            #else
            #    echo "Skipping $RUN_NAME (already exists)"
            #fi
        done
    done
done

