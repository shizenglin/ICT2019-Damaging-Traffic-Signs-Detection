#!/bin/sh
#SBATCH --partition=general --qos=short
#SBATCH --time=04:00:00
#SBATCH --mincpus=4
#SBATCH --mem=4000
#SBATCH --gres=gpu:pascal:1
#SBATCH --workdir=/home/nfs/tjviering
#SBATCH --job-name=traffic5
#SBATCH --output=/tudelft.net/staff-umbrella/deepgraphics/traffic_signs/cutout-master/logs/t5.out
#SBATCH --error=/tudelft.net/staff-umbrella/deepgraphics/traffic_signs/cutout-master/logs/t5.error
#SBATCH --mail-type=ALL

module use /opt/insy/modulefiles
module load cuda/9.0
module load cudnn/9.0-7.0.5

source pytorch3/bin/activate

cd /tudelft.net/staff-umbrella/deepgraphics/traffic_signs/cutout-master/david_code/

python train.py --train_dataset nl --test_dataset nl --experiment Snl7 --bam_dir ../datasets/BAM_data/ --gtsrb_dir ../datasets/GTSRB/Final_Training/Images/ --bam_conversion ../datasets/BAM_data/convention_conversion.csv --min_sqrt=32 --cuda=True --epochs=200 --damage_weight=7
