import numpy as np
# import tensorflow as tf
from tensorflow.python.framework import dtypes

## ivector :  Sampels X Dimension (2darray)
## labels : Samples (1darray)

class DataSet(object):
    
    def __init__(self,
                 ivectors,
                 labels,
                 dtype=dtypes.float32):
        
        self._ivectors = ivectors
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = ivectors.shape[0]
        self._dimension = ivectors.shape[1]
        
        
    @property
    def ivectors(self):
        return self._ivectors
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    @property
    def dimension(self):
        return self._dimension
    
    def next_batch(self,
                   batch_size,
                   shuffle):
        head = self._index_in_epoch
        
        # shuffling dataset at first batch of every epoch
        if head == 0 and shuffle:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._ivectors = self.ivectors[perm]
            self._labels = self.labels[perm]        

        # for last batch size => [total - batch_size : total]
        if head + batch_size > self._num_examples:
            self._index_in_epoch = self._num_examples - batch_size
            head = self._index_in_epoch
            
        # Last batch (reset index)
        if head + batch_size ==  self._num_examples:
            self._epochs_completed +=1
            tail = self._index_in_epoch + batch_size
            self._index_in_epoch = 0
            return self._ivectors[head:tail], self._labels[head:tail]            

        #normal batch
        else:
            self._index_in_epoch += batch_size
            tail = self._index_in_epoch
            return self._ivectors[head:tail], self._labels[head:tail]
        
        
        
           
        
        
        
        
    