#!/bin/bash
dataset='whu'
method='train'
data_root=/path/to/WHU
decoder_name='unet'
encoder_name='timm-resnest14d'
notes='bce_dice'

now=$(date +"%Y%m%d_%H%M%S")
exp=${decoder_name}'_'${encoder_name}'_'${notes}

config=configs/$dataset.yaml
train_id_path=$data_root/list/train.txt
test_id_path=$data_root/list/test.txt
val_id_path=$data_root/list/val.txt
save_path=checkpoints/whu

mkdir -p $save_path

cp ${method}.py $save_path
cp datasets/${dataset}.py $save_path

python ${method}.py --exp_name=$exp \
    --config=$config \
    --dataset=$dataset \
    --train-id-path $train_id_path \
    --test-id-path $test_id_path \
    --val_id_path $val_id_path \
    --save-path $save_path \
    --encoder_name $encoder_name \
    --decoder_name $decoder_name \
    --data_root=$data_root \
    --port 26800 2>&1 | tee $save_path/$now.log
