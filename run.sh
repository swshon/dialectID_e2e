#!/bin/bash

## Extract MFCC feature for NN input and save in tfrecords format: reading ./data/{train, dev, test}.txt
## example : prepare_data.py {feature type: logmel, mfcc, spec} {window_length} {window hop} {VAD} {CMVN: mv, m, False} {exclude short utterance under this frame number}
python ./scripts/prepare_data.py logmel 400 160 True mv 0
#exclude utterance under 2 seconds on training and dev dataset
python ./scripts/prepare_data.py logmel 400 160 True mv 200 

## Training NN with nn_model.py definition with original dataset
## example: train_e2e_custom.py {DNN_model_filename} {learning_rate} {input feature dimension} {featuretype} {total_step} {minibatch_size}
python ./scripts/train_e2e_custom.py e2e_model 0.001 40 False logmel_fft400_hop160_vad_cmvn 4000000 4
python ./scripts/train_e2e_custom.py e2e_model_exshort200_long 0.001 40 False logmel_fft400_hop160_vad_cmvn_exshort200 4000000 4


## Dataset augmentation using SOX command conrresponding speed(0.9, 1.0, 1.1) and volume (0.125, 1.0, 2.0 in amplitude)
mkdir -p data/wav/
for speed in 0.9 1.0 1.1; do
    for volume in 0.125 1.0 2.0; do
        python ./scripts/augmentation_by_speed__vol.py $speed $volume ./data/wav/
    done
done

## Listing augmented dataset files
find $(pwd)/data/wav/train/ -name '*.wav' >./data/train_speed_vol.txt
find $(pwd)/data/wav/dev/ -name '*.wav' >./data/dev_speed_vol.txt
find $(pwd)/data/wav/train/ -name '*v1.0*.wav' >./data/train_speed.txt
find $(pwd)/data/wav/dev/ -name '*v1.0*.wav' >./data/dev_speed.txt
find $(pwd)/data/wav/train/ -name 's1.0*.wav' >./data/train_vol.txt
find $(pwd)/data/wav/dev/ -name 's1.0*.wav' >./data/dev_vol.txt


## Extracting logmel feature of augmented dataset : reading ./data/{train, dev}_speed_vol.txt
## make sure same configuration as original feature extraction step
python ./scripts/prepare_augmented_data.py logmel 400 160 True mv 0
python ./scripts/prepare_augmented_data.py logmel 400 160 True mv 200

## Training NN iwth nn_model.py definition with augmented dataset
python ./scripts/train_e2e_custom.py e2e_model 0.001 40 False aug_logmel_fft400_hop160_vad_cmvn 4000000 4
python ./scripts/train_e2e_custom.py e2e_model_exshort200_long 0.001 40 False aug_logmel_fft400_hop160_vad_cmvn_exshort200 4000000 4









