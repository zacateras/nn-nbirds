import os
import random
import shutil
import cv2
import numpy as np
import pandas as pd
import re

def build_ds_meta():
    ds_meta_files = {
        'bounding_boxes': ['image_guid', 'x', 'y', 'xh', 'yh'],
        'classes': ['id', 'name'],
        'hierarchy': ['id', 'parent_id'],
        'image_class_labels': ['image_guid', 'class_id'],
        'images': ['image_guid', 'relative_path'],
        'photographers': ['image_guid', 'name'],
        'sizes': ['image_guid', 'width', 'height']
    }
    ds_meta = {}

    for ds_meta_file in ds_meta_files:
        with open ('data/%s.txt' % ds_meta_file, 'r' ) as f:
            content = f.read()
            
        for i in range(0, len(ds_meta_files[ds_meta_file]) - 1):
            content = re.sub('\ (.*)(\n|\Z)', r'|\1\2', content, flags = re.M)
        
        with open ('data/%s.csv' % ds_meta_file, 'w') as f:
            f.write(content)
        
        ds_meta[ds_meta_file] = pd.read_csv('data/%s.csv' % ds_meta_file, header=None, names=ds_meta_files[ds_meta_file], sep='|')
    
    return ds_meta

def guid_from_filename(filename):
    if len(filename) < 32:
        raise Exception('Provided filename %s is too short. Expected at least 32 characters.' % filename)
    
    return '%s-%s-%s-%s-%s' % (filename[0:8], filename[8:12], filename[12:16], filename[16:20], filename[20:32])
            
def apply_bounding_box(in_path, out_path, ds_meta, class_subdirs=True):
    bbs = ds_meta['bounding_boxes']
    
    subdirs = [''] if not class_subdirs else os.listdir(in_path)
    
    for subdir in subdirs:
        in_subdir_p = os.path.join(*(in_path, subdir))
        out_subdir_p = os.path.join(*(out_path, subdir))
        os.makedirs(out_subdir_p, exist_ok=True)
        
        for item in os.listdir(in_subdir_p):
            guid = guid_from_filename(item)
            img = cv2.imread(os.path.join(*(in_subdir_p, item)))
            bb = bbs[bbs['image_guid'] == guid]
            
            x = int(bb['x'])
            xh = int(bb['xh'])
            y = int(bb['y'])
            yh = int(bb['yh'])
            
            img_c = img[y:y+yh, x:x+xh]
            cv2.imwrite(os.path.join(*(out_subdir_p, item)), img_c)

def apply_gabor_filter(in_path, out_path, class_subdirs=True):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    
    subdirs = [''] if not class_subdirs else os.listdir(in_path)
    
    for subdir in subdirs:
        in_subdir_p = os.path.join(*(in_path, subdir))
        out_subdir_p = os.path.join(*(out_path, subdir))
        os.makedirs(out_subdir_p, exist_ok=True)
        
        for item in os.listdir(in_subdir_p):
            img = cv2.imread(os.path.join(*(in_subdir_p, item)))
            img_f = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
            cv2.imwrite(os.path.join(*(out_subdir_p, item)), img_f)

def apply_tvt_split(path, train=0.7, test=0.3, validation=0.0, class_subdirs=True):
    dir_train_p = '%s_train' % path
    dir_validation_p = '%s_validation' % path
    dir_test_p = '%s_test' % path
    
    if os.path.exists(dir_train_p) and os.path.isdir(dir_train_p):
        shutil.rmtree(dir_train_p)
    if os.path.exists(dir_validation_p) and os.path.isdir(dir_validation_p):
        shutil.rmtree(dir_validation_p)
    if os.path.exists(dir_test_p) and os.path.isdir(dir_test_p):
        shutil.rmtree(dir_test_p)
        
    subdirs = [''] if not class_subdirs else os.listdir(path)
    
    for subdir in subdirs:
        subdir_p = os.path.join(*(path, subdir))
        subdir_list = os.listdir(subdir_p)
        
        random.shuffle(subdir_list)
        
        subdir_list_len = len(subdir_list)
        subdir_list_train_thld = int(subdir_list_len * train)
        subdir_list_validation_thld = int(subdir_list_len * validation) + subdir_list_train_thld
        
        subdir_list_train = subdir_list[:subdir_list_train_thld]
        subdir_list_validation = subdir_list[subdir_list_train_thld:subdir_list_validation_thld]
        subdir_list_test = subdir_list[subdir_list_validation_thld:]
        
        # prepare train part
        subdir_train_p = os.path.join(*(dir_train_p, subdir))
        os.makedirs(subdir_train_p, exist_ok=True)
        for subdir_list_train_item in subdir_list_train:
            src = os.path.join(subdir_p, subdir_list_train_item)
            dest = os.path.join(subdir_train_p, subdir_list_train_item)
            shutil.copyfile(src, dest)
            
        # prepare validation part
        if len(subdir_list_validation) > 0:
            subdir_validation_p = os.path.join(*(dir_validation_p, subdir))
            os.makedirs(subdir_validation_p, exist_ok=True)
            for subdir_list_validation_item in subdir_list_validation:
                src = os.path.join(subdir_p, subdir_list_validation_item)
                dest = os.path.join(subdir_validation_p, subdir_list_validation_item)
                shutil.copyfile(src, dest)
            
        # prepare test part
        subdir_test_p = os.path.join(*(dir_test_p, subdir))
        os.makedirs(subdir_test_p, exist_ok=True)
        for subdir_list_test_item in subdir_list_test:
            src = os.path.join(subdir_p, subdir_list_test_item)
            dest = os.path.join(subdir_test_p, subdir_list_test_item)
            shutil.copyfile(src, dest)