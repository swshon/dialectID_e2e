import numpy as np
import librosa
import sys
import tensorflow as tf

def label_indexer(label_text):
    return {
        'LAV': 2,
        'EGY': 0,
        'MSA': 3,
        'NOR': 4,
        'GLF': 1,
        
    }[label_text]

def cmvn_slide(feat,winlen=300,cmvn=False): #feat : (length, dim) 2d matrix
    maxlen = np.shape(feat)[0]
    new_feat = np.empty_like(feat)
    cur = 1
    leftwin = 0
    rightwin = winlen/2
    
    # middle
    for cur in range(maxlen):
        cur_slide = feat[cur-leftwin:cur+rightwin,:] 
        #cur_slide = feat[cur-winlen/2:cur+winlen/2,:]
        mean =np.mean(cur_slide,axis=0)
        std = np.std(cur_slide,axis=0)
        if cmvn == 'mv':
            new_feat[cur,:] = (feat[cur,:]-mean)/std # for cmvn        
        elif cmvn =='m':
            new_feat[cur,:] = (feat[cur,:]-mean) # for cmn
        if leftwin<winlen/2:
            leftwin+=1
        elif maxlen-cur < winlen/2:
            rightwin-=1    
    return new_feat

def feat_extract(filename,feat_type,n_fft_length=512,hop=160,vad=True,cmvn=False,exclude_short=500):
    filelist = np.loadtxt(filename,delimiter='\t',dtype='string',usecols=(0))
#     utt_label = np.loadtxt(filename,delimiter='\t',dtype='int',usecols=(1))
    
    feat = []
    utt_shape = []
    new_utt_label =[]
    for index,wavname in enumerate(filelist):
        #read audio input
        y, sr = librosa.core.load(wavname,sr=16000,mono=True,dtype='float')
        temp_label = label_indexer( wavname.split('/')[-2])

        #extract feature
        if feat_type =='melspec':
            Y = librosa.feature.melspectrogram(y,sr,n_fft=n_fft_length,hop_length=hop,n_mels=40,fmin=133,fmax=6955)
        elif feat_type =='mfcc':
            Y = librosa.feature.mfcc(y,sr,n_fft=n_fft_length,hop_length=hop,n_mfcc=40,fmin=133,fmax=6955)
        elif feat_type =='spec':
            Y = np.abs( librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400) )
        elif feat_type =='logspec':
            Y = np.log( np.abs( librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400) ) )
        elif feat_type =='logmel':
            Y = np.log( librosa.feature.melspectrogram(y,sr,n_fft=n_fft_length,hop_length=hop,n_mels=40,fmin=133,fmax=6955) )

        Y = Y.transpose()

        # Simple VAD based on the energy
        if vad:
            E = librosa.feature.rmse(y, frame_length=n_fft_length,hop_length=hop,)
            threshold= np.mean(E)/2 * 1.04
            vad_segments = np.nonzero(E>threshold)
            if vad_segments[1].size!=0:
                Y = Y[vad_segments[1],:]

                
        #exclude short utterance under "exclude_short" value
        if exclude_short == 0 or (Y.shape[0] > exclude_short):
            if cmvn:
                Y = cmvn_slide(Y,300,cmvn)
            feat.append(Y)
            utt_shape.append(np.array(Y.shape))
            new_utt_label.append(temp_label)
            sys.stdout.write('%s\r' % index)
            sys.stdout.flush()

        
    tffilename = feat_type+'_fft'+str(n_fft_length)+'_hop'+str(hop)
    if vad:
        tffilename += '_vad'
    if cmvn=='m':
        tffilename += '_cmn'
    elif cmvn =='mv':
        tffilename += '_cmvn'
    if exclude_short >0:
        tffilename += '_exshort'+str(exclude_short)

    return feat, new_utt_label, utt_shape, tffilename #feat : (length, dim) 2d matrix


def write_tfrecords(feat, utt_label, utt_shape, tfrecords_name):
    writer = tf.python_io.TFRecordWriter(tfrecords_name)
    trIdx = range(np.shape(utt_label)[0])
    
    # iterate over each example
    # wrap with tqdm for a progress bar
    for count,idx in enumerate(trIdx):
        feats = feat[idx].reshape(feat[idx].size)
        label = utt_label[idx]
        shape = utt_shape[idx]

        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(     
              # Features contains a map of string to Feature proto objects
              feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'labels': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])),
                'shapes': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=shape)),
                'features': tf.train.Feature(
                    float_list=tf.train.FloatList(value=feats.astype("float32"))),
        }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)

    writer.close()
    print tfrecords_name+": total "+str(len(feat))+" feature and "+str(np.shape(utt_label))+" label saved"
    
    
def do_shuffle(feat,utt_label,utt_shape):
    
    #### shuffling
    shuffleidx = np.arange(0,len(feat))
    np.random.shuffle(shuffleidx)

    feat=np.array(feat)
    utt_label=np.array(utt_label)
    utt_shape = np.array(utt_shape)

    feat = feat[shuffleidx]
    utt_label = utt_label[shuffleidx]
    utt_shape = utt_shape[shuffleidx]

    feat = feat.tolist()
    utt_label = utt_label.tolist()
    utt_shape = utt_shape.tolist()
    
    return feat, utt_label, utt_shape


####file load & extracting mel-spectrogram
FEAT_TYPE = 'mfcc' #mfcc or melspec
N_FFT = 512
HOP = 160
EXCLUDE_SHORT=500
VAD = True
CMVN = 'm'


if len(sys.argv)>1:
    if len(sys.argv)< 7:
        print "not enough arguments"
        print "example : python step05_prepare_augmented_data_speed.py mfcc 512 160 True mv 500"

    FEAT_TYPE = sys.argv[1]
    N_FFT = int(sys.argv[2])
    HOP = int(sys.argv[3])
    VAD = sys.argv[4]
    CMVN = sys.argv[5]
    EXCLUDE_SHORT = int(sys.argv[6])
    
    
if VAD =='False':
    VAD = False
if CMVN == 'False':
    CMVN = False

FILENAME = 'data/train_speed.txt'
feat, utt_label, utt_shape, tffilename = feat_extract(FILENAME,FEAT_TYPE,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT)
print tffilename

TOTAL_JOB = 1
for i in range(TOTAL_JOB):
#     TFRECORDS_NAME = 'data/tfrecords/'+'mgb3_aug_'+ tffilename +'_train.'+str(i)+'.tfrecords'
#     write_tfrecords(feat,utt_label,utt_shape,TFRECORDS_NAME)
    feat, utt_label, utt_shape = do_shuffle(feat,utt_label,utt_shape)
    TFRECORDS_NAME = 'data/tfrecords/'+'mgb3_aug_speed_'+tffilename+'_train_shuffle.'+str(i)+'.tfrecords'
    write_tfrecords(feat,utt_label,utt_shape,TFRECORDS_NAME)
    
    
FILENAME = 'data/dev_speed.txt'
feat, utt_label, utt_shape, tffilename = feat_extract(FILENAME,FEAT_TYPE,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT)
print tffilename

TOTAL_JOB = 1
for i in range(TOTAL_JOB):
#     TFRECORDS_NAME = 'data/tfrecords/'+'mgb3_aug_'+ tffilename +'_dev.'+str(i)+'.tfrecords'
#     write_tfrecords(feat,utt_label,utt_shape,TFRECORDS_NAME)
    feat, utt_label, utt_shape = do_shuffle(feat,utt_label,utt_shape)
    TFRECORDS_NAME = 'data/tfrecords/'+'mgb3_aug_speed_'+tffilename+'_dev_shuffle.'+str(i)+'.tfrecords'
    write_tfrecords(feat,utt_label,utt_shape,TFRECORDS_NAME)
