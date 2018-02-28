# End-to-end Dialect Identification (implementation on MGB-3 Arabic dialect dataset)
Tensorflow implementation of End-to-End dialect identificaion in Arabic. If you are familiar with Language/Speaker identificatio/verification, it can be easily modified to another dialect, language or even speaker identification/verification tasks.

# Requirment
* Python, tested on 2.7.6
* Tensorflow > v1.0
* python library > sox, tested on 1.3.2
* python library > librosa, tested on 0.5.1 

# Data list format
datalist consist of (location of wavfile) and (label in digit).

Example) "train.txt"
```
./data/wav/EGY/EGY000001.wav 0
./data/wav/EGY/EGY000002.wav 0
./data/wav/NOR/NOR000001.wav 4
```

Labels of Dialect: 
- Egytion (EGY) : 0
- Gulf (GLF) : 1
- Levantine(LAV): 2
- Modern Standard Arabic (MSA) : 3
- North African (NOR): 4

# Dataset Augmentation
Augementation was done by two different method. First is random segment of the input utterance, and the other is perturbation by modifying speed and volume of speech.



# Model definition
Simple description of the DNN model:
![Image of Model](https://github.com/swshon/dialectID_e2e/blob/master/images/figure_network.png)
we used four 1-dimensional CNN (1d-CNN) layers (40x5 - 500x7 - 500x1 - 500x1 filter sizes with 1-2-1-1 strides and the number of filters is 500-500-500-3000) and two FC layers (1500-600) that are connected with a Global average pooling layer which averages the CNN outputs to produce a fixed output size of 3000x1. 

End-to-end DID accuracy by epoch
![Image of Model](https://github.com/swshon/dialectID_e2e/blob/master/images/accuracy_aug.png)
End-to-end DID accuracy by epoch using augmented dataset
![Image of Model](https://github.com/swshon/dialectID_e2e/blob/master/images/accuracy_feat.png)
Performance comparison with and without Random Segmentation(RS)
![Image of Model](https://github.com/swshon/dialectID_e2e/blob/master/images/random_segment.png)


# Performance evaluation 
Best performance is 73.39% on Accuracy. (Feb.28 2018)

for reference,

Conventional i-vector with SVM : 60.32%<br />
Conventional i-vector with LDA and Cosine Distance : 62.60%<br />
End-to-End model without dataset augmentation(MFCC): 65.55%<br />
End-to-End model without dataset augmentation(FBANK): 64.81%<br />
End-to-End model without dataset augmentation(Spectrogram): 57.57%<br />

End-to-End model with volume perturbation(MFCC) : 67.49%<br />
End-to-End model with speed perturbation(MFCC) : 70.51%<br />

End-to-End model with speed and volume perturbation (MFCC) : 70.91%<br />
End-to-End model with speed and volume perturbation (FBANK) : 71.92%<br />
End-to-End model with speed and volume perturbation (Spectrogram) : 68.83%<br />

End-to-End model with speed and volume perturbation+random segmention (MFCC) : 71.05%<br />
End-to-End model with speed and volume perturbation+random segmention (FBANK) : 73.39%<br />
End-to-End model with speed and volume perturbation+random segmention (Spectrogram) : 70.17%<br />


# Offline test
Offline test can be done in offline_test.ipynb code on our pretrained model. Specify wav file you want to identify Arabic dialect by modifying FILENAME variable.

```
FILENAME = ['/data/test/EGY_00001.wav']
```

Result can be shown like below bar plot of likelihood on 5 Arabic dialects.

![Image of offline result plot](https://github.com/swshon/dialectID_e2e/blob/master/images/offline_plot.png)


