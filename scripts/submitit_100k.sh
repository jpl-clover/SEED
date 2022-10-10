#!/bin/bash
current_file_dir=$(realpath $(dirname $0))
cd "$current_file_dir/.."  # this is the root of the SEED repo
# Change these paths to your own
msl_dir=$HOME/clover_shared/datasets/msl_images
simclr_weights_dir=$HOME/clover_shared/simclr_weights/pretrained
model_path_root=/scratch/07265/egoh/Documents/CLOVER/distillation_100k

timestamp=$(date +%Y%m%d_%H%M%S)

teacher_archs=("r50_1x_sk0" "r50_2x_sk1" "r101_1x_sk0" "r101_2x_sk1" "r152_1x_sk0" "r152_2x_sk1")
student_archs=("seed_efficientnet_b0" "mobilenetv3_large" "seed_efficientnet_b1" "resnet18" "resnet50")
for teacher_arch in "${teacher_archs[@]}"; do
    for student_arch in "${student_archs[@]}"; do
        echo "Submitting $teacher_arch -> $student_arch"
        python main_submitit.py \
            --config config/isaac/msl_100k_config.yaml \
            --teacher_arch $teacher_arch \
            --teacher_weights $simclr_weights_dir/$teacher_arch.pth \
            --student_arch $student_arch \
            --print_freq 20 \
            --model_path $model_path_root/$teacher_arch"_"$student_arch"_"$timestamp \
            --dataset_dir $msl_dir &
        echo $! >> $current_file_dir/job_ids_$timestamp.txt  # save the job id so we can kill them later if necessary
        sleep $((RANDOM % 10))  # prevent overloading Longhorn
    done
done

cd "$current_file_dir"