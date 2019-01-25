python train.py --train_dataset de --test_dataset nl_nl --experiment de3 --bam_dir ../../../datasets/NL/ --gtsrb_dir ../../../datasets/GTSRB_data/Final_Training/Images/ --bam_conversion ../../../datasets/NL/convention_conversion.csv --min_sqrt=32 --cuda=True --epochs=200 --damage_weight=1
python train.py --train_dataset nl --test_dataset nl --experiment nl3 --bam_dir ../../../datasets/NL/ --gtsrb_dir ../../../datasets/GTSRB_data/Final_Training/Images/ --bam_conversion ../../../datasets/NL/convention_conversion.csv --min_sqrt=32 --cuda=True --epochs=200 --damage_weight=1
python train.py --train_dataset de_nl --test_dataset nl --experiment de_nl3 --bam_dir ../../../datasets/NL/ --gtsrb_dir ../../../datasets/GTSRB_data/Final_Training/Images/ --bam_conversion ../../../datasets/NL/convention_conversion.csv --min_sqrt=32 --cuda=True --epochs=200 --damage_weight=1

python train.py --train_dataset de --test_dataset nl_nl --experiment de4 --bam_dir ../../../datasets/NL/ --gtsrb_dir ../../../datasets/GTSRB_data/Final_Training/Images/ --bam_conversion ../../../datasets/NL/convention_conversion.csv --min_sqrt=32 --cuda=True --epochs=200 --damage_weight=7
python train.py --train_dataset nl --test_dataset nl --experiment nl4 --bam_dir ../../../datasets/NL/ --gtsrb_dir ../../../datasets/GTSRB_data/Final_Training/Images/ --bam_conversion ../../../datasets/NL/convention_conversion.csv --min_sqrt=32 --cuda=True --epochs=200 --damage_weight=7
python train.py --train_dataset de_nl --test_dataset nl --experiment de_nl4 --bam_dir ../../../datasets/NL/ --gtsrb_dir ../../../datasets/GTSRB_data/Final_Training/Images/ --bam_conversion ../../../datasets/NL/convention_conversion.csv --min_sqrt=32 --cuda=True --epochs=200 --damage_weight=7



