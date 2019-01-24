# !/bin/bash

bam_dir=./BAM;
gtsrb_dir=./GTSRB/Final_Training/Images;
bam_conversion=./BAM/convention_conversion.csv;

python train.py --bam_dir $bam_dir --gtsrb_dir $gtsrb_dir --bam_conversion $bam_conversion \
                --batch_size 128 --epochs 200 --cuda True \
                --min_sqrt 32 \
                --train_dataset nl --test_dataset nl \
                --damage_weights 1 1 \
                --img_size 32 \
                --experiment nl/nl/imsize=32/a=1/w=1/minsqrt=32 \
                --alpha 1 &


python train.py --bam_dir $bam_dir --gtsrb_dir $gtsrb_dir --bam_conversion $bam_conversion \
                --batch_size 128 --epochs 200 --cuda True \
                --min_sqrt 32 \
                --train_dataset de --test_dataset nl_nl \
                --damage_weights 1 1 \
                --img_size 32 \
                --experiment de/nl_nl/imsize=32/a=1/w=1/minsqrt=32 \
                --alpha 1 &


python train.py --bam_dir $bam_dir --gtsrb_dir $gtsrb_dir --bam_conversion $bam_conversion \
                --batch_size 128 --epochs 200 --cuda True \
                --min_sqrt 32 \
                --train_dataset de_nl --test_dataset nl \
                --damage_weights 1 1 \
                --img_size 32 \
                --experiment de_nl/nl/imsize=32/a=1/w=1/minsqrt=32 \
                --alpha 1
