#!/data/sls/u/swshon/tools/pytf/bin/python
import os,sys
import tensorflow as tf
import numpy as np
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')
from tensorflow.contrib.learn.python.learn.datasets import base



def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    true_class = np.argmax(labels, 1)
#     print pred_class
#     print true_class
    return (100.0 * np.sum(pred_class == true_class) / predictions.shape[0])

def txtwrite(filename, dict):
    with open(filename, "w") as text_file:
        for key, vec in dict.iteritems():
            text_file.write('%s [' % key)
            for i, ele in enumerate(vec):
                text_file.write(' %f' % ele)
            text_file.write(' ]\n')

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
            
        
#### function for read tfrecords
def read_and_decode_emnet_mfcc(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer(filename, name = 'queue')
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'labels': tf.FixedLenFeature([], tf.int64),
            'shapes': tf.FixedLenFeature([2], tf.int64),
            'features': tf.VarLenFeature( tf.float32)
        })
    # now return the converted data
    labels = features['labels']
    shapes = features['shapes']
    feats = features['features']
    shapes = tf.cast(shapes, tf.int32)
    feats2d = tf.reshape(feats.values, shapes)
    feats1d = feats.values
    return labels, shapes, feats2d


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

### Variable Initialization    
NUMGPUS = 1
BATCHSIZE = 4
ITERATION = 4000000
SAVE_INTERVAL = 4000
LOSS_INTERVAL = 100
TESTSET_INTERVAL = 2000
MAX_SAVEFILE_LIMIT = 1000
# DATASET_NAME = 'post_pooled/train_gmm.h5'
# DURATION_LIMIT = 1000 #(utterance below DURATION_LIMIT/100 seconds will be mismissed :default=1000)
# SPKCOUNT_LIMIT = 3 #(speaker with equal or less than this number will be dismissed :default=3)
# MIXTURE = 2048  # = number of softmax layer
# filelist = ['../post_pooled/swb_gmm','../post_pooled/sre_gmm'] # dataset for training
# post_mean = np.empty((0,MIXTURE),dtype='float32')
# post_std = np.empty((0,MIXTURE),dtype='float32')
utt_label = []
duration = np.empty(0,dtype='int')
spklab = []
TFRECORDS_FOLDER = '/data/sls/scratch/swshon/exp_scratch/dialectID_e2e_spectrum/data/tfrecords/'
SAVER_FOLDERNAME = 'saver'

if len(sys.argv)< 7:
    print "not enough arguments"
    print "command : ./new_training.py [nn_model_name] [learning rate] [input_dim(feat dim)] [is_batch_norm] [feature_filename]"
    print "(example) ./new_training.py nn_model 0.001 40 True aug_mfcc_fft512_hop160_vad_cmn"

is_batchnorm = False
NN_MODEL = sys.argv[1]
LEARNING_RATE = np.float(sys.argv[2])
INPUT_DIM = sys.argv[3]
IS_BATCHNORM = sys.argv[4]
FEAT_TYPE = sys.argv[5]
ITERATION = int(sys.argv[6])
BATCHSIZE = int(sys.argv[7])
#NN_MODEL = 'new_nn_model'
#LEARNING_RATE = 0.001
#INPUT_DIM = 40
#IS_BATCHNORM = True
#FEAT_TYPE = 'mfcc_fft512_hop160_vad_cmn'

SAVER_FOLDERNAME = 'saver/'+NN_MODEL+'_'+FEAT_TYPE
if IS_BATCHNORM=='True':
    SAVER_FOLDERNAME = SAVER_FOLDERNAME + '_BN'
    is_batchnorm = True
nn_model = __import__(NN_MODEL)

# records_list = []
# for i in range(0,1):
#     records_list.append(TFRECORDS_FOLDER+'mgb3_train.'+str(i)+'.tfrecords')
records_shuffle_list = []
for i in range(0,1):
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_train_shuffle.'+str(i)+'.tfrecords')
records_dev_shuffle_list = []
for i in range(0,1):
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')
    records_shuffle_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_dev_shuffle.'+str(i)+'.tfrecords')


labels,shapes,feats = read_and_decode_emnet_mfcc(records_shuffle_list)
labels_batch,feats_batch,shapes_batch = tf.train.batch(
    [labels, feats,shapes], batch_size=BATCHSIZE, dynamic_pad=True, allow_smaller_final_batch=True,
    capacity=50)
#FEAT_TYPE = 'mfcc_fft512_hop160_vad_cmn'
FEAT_TYPE = FEAT_TYPE.split('_exshort')[0]
FEAT_TYPE = FEAT_TYPE.split('aug_')[-1]
FEAT_TYPE = FEAT_TYPE.split('vol_')[-1]
FEAT_TYPE = FEAT_TYPE.split('speed_')[-1]
records_test_list = []
for i in range(0,1):
    records_test_list.append(TFRECORDS_FOLDER+'mgb3_'+FEAT_TYPE+'_test_shuffle.'+str(i)+'.tfrecords')


#data for validation
vali_labels,vali_shapes,vali_feats = read_and_decode_emnet_mfcc(records_test_list)
vali_labels_batch,vali_feats_batch,vali_shapes_batch = tf.train.batch(
    [vali_labels, vali_feats, vali_shapes], batch_size=BATCHSIZE, dynamic_pad=True, allow_smaller_final_batch=True,
    capacity=50,num_threads=1)



### Initialize network related variables

with tf.device('/cpu:0'):

    softmax_num = 5
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                               50000, 0.98, staircase=True)

    opt = tf.train.GradientDescentOptimizer(learning_rate)

    emnet_losses = []
    emnet_grads = []

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(NUMGPUS):
            with tf.device('/gpu:%d' % i):
                emnet = nn_model.nn(feats_batch, labels_batch,labels_batch, shapes_batch, softmax_num,True,INPUT_DIM,is_batchnorm)
                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(emnet.loss)
                emnet_losses.append(emnet.loss)
                emnet_grads.append(grads)
        
        with tf.device('/gpu:0'):
            emnet_validation = nn_model.nn(vali_feats_batch,vali_labels_batch,vali_labels_batch,vali_shapes_batch, softmax_num,False,INPUT_DIM,is_batchnorm);
            tf.get_variable_scope().reuse_variables()
    
    loss = tf.reduce_mean(emnet_losses)        
    grads = average_gradients(emnet_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=MAX_SAVEFILE_LIMIT)


    summary_writer = tf.summary.FileWriter(SAVER_FOLDERNAME, sess.graph)
    #variable_summaries(loss)

    tf.summary.scalar('loss', loss)
    tf.summary.histogram('loss', loss)
    acc_tf = tf.Variable(tf.zeros([1]),name='acc')
    tf.summary.scalar('Accuracy_tst', tf.squeeze(acc_tf))

    summary_op = tf.summary.merge_all()
    
    tf.initialize_all_variables().run()
    tf.train.start_queue_runners(sess=sess)
    ### Training neural network 
    resume=False
    START=0
    if resume:
        START=1000000
        saver.restore(sess,SAVER_FOLDERNAME+'/model'+str(START)+'.ckpt')

    
    for step in range(START,ITERATION):

        _, loss_v,mean_loss = sess.run([apply_gradient_op, emnet.loss,loss])
        
        
        if np.isnan(loss_v):
            print ('Model diverged with loss = NAN')
            quit()


        if step % TESTSET_INTERVAL ==0 and step >=START:
            prediction = np.empty((0,5))
            _label = np.int64([])
            for iter in range((1492/BATCHSIZE) +1):
                pre,lab = sess.run([emnet_validation.o1, emnet_validation.label ])
                prediction = np.append(prediction,pre,axis=0)
                _label = np.append(_label, lab)
            prediction = prediction[0:1492,:]
            _label = _label[0:1492]            
#             prediction, _label= sess.run([emnet_validation.o1, emnet_validation.label ])
            spklab_num_mat = np.eye(softmax_num)[_label] 
            acc = accuracy(prediction, spklab_num_mat)
            print ('Step %d: loss %.6f, lr : %.5f, Accuracy : %f' % (step,mean_loss, sess.run(learning_rate),acc))
            acc_op = acc_tf.assign([acc])

        if step % LOSS_INTERVAL ==0:
            print ('Step %d: loss %.6f, lr : %.5f' % (step, mean_loss, sess.run(learning_rate)))
            summary_str, _ = sess.run([summary_op, acc_op])
            summary_writer.add_summary(summary_str,step)

        if step % SAVE_INTERVAL == 0 and step >START:
            saver.save(sess, SAVER_FOLDERNAME+'/model'+str(step)+'.ckpt',global_step=step)
