import sox
import sys
import numpy as np


def creat_vol_augmentation(filename, target_folder, vol_list):
    aug_generator = sox.Transformer()
    for volume in vol_list:
        aug_generator.vol(volume)
        filelist = np.loadtxt(filename,delimiter='\t',dtype='string',usecols=(0))
        for index,files in enumerate(filelist):
            save_filename = TARGET_FOLDER+filename.split('.')[0].split('/')[1] + '/'+ files.split('/')[-2]+'/v'+str(volume)+'_'+files.split('/')[-1]
            aug_generator.build(files,save_filename)

def creat_speed_augmentation(filename, target_folder, speed_list):
    aug_generator = sox.Transformer()
    for speed in speed_list:
        aug_generator.speed(speed)
        filelist = np.loadtxt(filename,delimiter='\t',dtype='string',usecols=(0))
        for index,files in enumerate(filelist):
            save_filename = TARGET_FOLDER+filename.split('.')[0].split('/')[1] + '/'+ files.split('/')[-2]+'/s'+str(speed)+'_'+files.split('/')[-1]
            aug_generator.build(files,save_filename)
        

def creat_speed_vol_augmentation(filename, target_folder, speed, volume):
    aug_generator = sox.Transformer()
    aug_generator.vol(volume)
    aug_generator.speed(speed)
    filelist = np.loadtxt(filename,delimiter='\t',dtype='string',usecols=(0))
    for index,files in enumerate(filelist):
        save_filename = TARGET_FOLDER+filename.split('.')[0].split('/')[1] + '/'+ files.split('/')[-2]+'/s'+str(speed)+'_v'+str(volume)+'_'+files.split('/')[-1]
        aug_generator.build(files,save_filename)
        
        
SPEED_LIST = [0.9, 1.0, 1.1]
VOL_LIST = [0.125, 1.0, 2.0]
TARGET_FOLDER = './data/wav/'


if len(sys.argv)>1:
    if len(sys.argv)< 4:
        print "not enough arguments"
        print "usage : python augmentation_by_speed_vol.py [speed] [vol] [target folder]"
        print "example : python augmentation_by_speed_vol.py 0.9 0.125 ./data/wav/"
        
    SPEED_LIST = np.float(sys.argv[1])
    VOL_LIST = np.float(sys.argv[2])
    TARGET_FOLDER = './data/wav/'


FILENAME_LIST = ['data/train.txt', 'data/dev.txt', 'data/test.txt']
for filename in FILENAME_LIST:
    print 'processing'+filename+'....'
    creat_speed_vol_augmentation(filename, TARGET_FOLDER, SPEED_LIST, VOL_LIST)
#     creat_speed_augmentation(filename, TARGET_FOLDER, SPEED_LIST)
#     creat_vol_augmentation(filename, TARGET_FOLDER, VOL_LIST)

