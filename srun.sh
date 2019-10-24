#!/bin/bash

# Extract MFCC feature for NN input and save in tfrecords format
#./scripts/prepare_data.py mfcc 400 160 True m 0
#./scripts/prepare_data.py melspec 400 160 True m 0
#./scripts/prepare_data.py spec 400 160 True m 0
#./scripts/prepare_data.py melspec 400 160 True m 200
#./scripts/prepare_data.py mfcc 400 160 True m 200
#./scripts/prepare_data.py spec 400 160 True m 200

# Training NN with nn_model.py definition with original dataset
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_mfcc_fft400_hop160_vad_cmn.out ./scripts/train_e2e.py new_nn_model 0.001 40 False mfcc_fft400_hop160_vad_cmn &
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_melspec_fft400_hop160_vad_cmn.out ./scripts/train_e2e.py new_nn_model 0.001 40 False melspec_fft400_hop160_vad_cmn &
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_spec_fft400_hop160_vad_cmn.out ./scripts/train_e2e.py new_nn_model 0.001 40 False spec_fft400_hop160_vad_cmn &
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_exshort200_long_mfcc_fft400_hop160_vad_cmn.out ./scripts/train_e2e.py new_nn_model_exshort200_long 0.001 40 False mfcc_fft400_hop160_vad_cmn_exshort200 &
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_exshort200_long_melspec_fft400_hop160_vad_cmn.out ./scripts/train_e2e.py new_nn_model_exshort200_long 0.001 40 False melspec_fft400_hop160_vad_cmn_exshort200 &
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_exshort200_long_spec_fft400_hop160_vad_cmn.out ./scripts/train_e2e.py new_nn_model_exshort200_long 0.001 40 False spec_fft400_hop160_vad_cmn_exshort200 &


# Dataset augmentation using SOX command conrresponding speed(0.9, 1.0, 1.1) and volume (0.125, 1.0, 2.0 in amplitude)
#./scripts/augmentation_by_speed_vol.py 0.9 0.125 ./data/wav/
#./scripts/augmentation_by_speed_vol.py 1.0 0.125 ./data/wav/
#./scripts/augmentation_by_speed_vol.py 1.1 0.125 ./data/wav/
#./scripts/augmentation_by_speed_vol.py 0.9 1.0 ./data/wav/
#./scripts/augmentation_by_speed_vol.py 1.0 1.0 ./data/wav/
#./scripts/augmentation_by_speed_vol.py 1.1 1.0 ./data/wav/
#./scripts/augmentation_by_speed_vol.py 0.9 2.0 ./data/wav/
#./scripts/augmentation_by_speed_vol.py 1.0 2.0 ./data/wav/
#./scripts/augmentation_by_speed_vol.py 1.1 2.0 ./data/wav/

# Listing augmented dataset files
#find $(pwd)/data/wav/train/ -name '*.wav' >./data/train_speed_vol.txt
#find $(pwd)/data/wav/dev/ -name '*.wav' >./data/dev_speed_vol.txt
#find $(pwd)/data/wav/test/ -name '*.wav' >./data/test_speed_vol.txt
#find $(pwd)/data/wav/train/ -name '*v1.0*.wav' >./data/train_speed.txt &
#find $(pwd)/data/wav/dev/ -name '*v1.0*.wav' >./data/dev_speed.txt &
#find $(pwd)/data/wav/train/ -name 's1.0*.wav' >./data/train_vol.txt &
#find $(pwd)/data/wav/dev/ -name 's1.0*.wav' >./data/dev_vol.txt &

# Extracting MFCC feature of augmented dataset
srun --partition=630 --cpus-per-task=2 --mem=20GB --out=./log/aug_mfcc_fft400_hop160_vad_cmn ./scripts/prepare_augmented_data.py mfcc 400 160 True m 0 &
srun --partition=630 --cpus-per-task=2 --mem=20GB --out=./log/aug_mfcc_fft400_hop160_vad_cmn ./scripts/prepare_augmented_data.py mfcc 400 160 True m 200 &
srun --partition=630 --cpus-per-task=2 --mem=20GB --out=./log/aug_mfcc_fft400_hop160_vad_cmn ./scripts/prepare_augmented_data_vol.py mfcc 400 160 True m 0 &
srun --partition=630 --cpus-per-task=2 --mem=20GB --out=./log/aug_mfcc_fft400_hop160_vad_cmn ./scripts/prepare_augmented_data_vol.py mfcc 400 160 True m 200 &
srun --partition=630 --cpus-per-task=2 --mem=20GB --out=./log/aug_mfcc_fft400_hop160_vad_cmn ./scripts/prepare_augmented_data_speed.py mfcc 400 160 True m 0 &
srun --partition=630 --cpus-per-task=2 --mem=20GB --out=./log/aug_mfcc_fft400_hop160_vad_cmn ./scripts/prepare_augmented_data_speed.py mfcc 400 160 True m 200 &


# Training NN iwth nn_model.py definition with augmented dataset
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_aug_mfcc_fft400_hop160_vad_cmn.out ./scripts/train_e2e.py new_nn_model 0.001 40 False aug_mfcc_fft400_hop160_vad_cmn &
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_aug_mfcc_fft400_hop160_vad_cmn_exshort.out ./scripts/train_e2e.py new_nn_model_exshort200_long 0.001 40 False aug_mfcc_fft400_hop160_vad_cmn_exshort200 &

srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_aug_vol_mfcc_fft400_hop160_vad_cmn.out ./scripts/train_e2e.py new_nn_model 0.001 40 False aug_vol_mfcc_fft400_hop160_vad_cmn &
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_aug_vol_mfcc_fft400_hop160_vad_cmn_exshort.out ./scripts/train_e2e.py new_nn_model_exshort200_long 0.001 40 False aug_vol_mfcc_fft400_hop160_vad_cmn_exshort200 &

srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_aug_speed_mfcc_fft400_hop160_vad_cmn.out ./scripts/train_e2e.py new_nn_model 0.001 40 False aug_speed_mfcc_fft400_hop160_vad_cmn &
srun --partition=titanx --gres=gpu:1 --cpus-per-task=2 --mem=16GB --output=./log/new_nn_model_aug_speed_mfcc_fft400_hop160_vad_cmn_exshort.out ./scripts/train_e2e.py new_nn_model_exshort200_long 0.001 40 False aug_speed_mfcc_fft400_hop160_vad_cmn_exshort200 &





## Extract framelevel embedding
srun -p sm --gres=gpu:1 --output=log/test.log python ./scripts/extract_framelevel_embeddings.py --wavlist data/test.txt --outputlayer &
srun -p sm --gres=gpu:1 --output=log/dev.log python ./scripts/extract_framelevel_embeddings.py --wavlist data/dev.txt --outputlayer &
srun -p sm --gres=gpu:1 --output=log/train.log python ./scripts/extract_framelevel_embeddings.py --wavlist data/train.txt --outputlayer &

