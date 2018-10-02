"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python train.py --weight imagenet --logs PATH/TO/LOGS --dataset PATH/TO/DATASET --subset train


"""

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import json
import datetime
import time
import numpy as np
import random
import skimage.io
from imgaug import augmenters as iaa
import numexpr as ne

import pandas as pd
import pygsheets



# Import Mask RCNN 
# MASKRCNN_CODE_PATH = "/home/jimmy15923/model_zoo/keras_maskrcnn/"
# sys.path.append(MASKRCNN_CODE_PATH)  # To find local version of the library

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.model import *
import monuconfig

# Path to trained weights file

## download path to coco.h5 from https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

COCO_WEIGHTS_PATH = "PATH/TO/COCO.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = 'logs'

# set seed for same validation split
seed = 7
IMAGE_IDS = next(os.walk('dataset/train'))[1]
random.seed(seed)
random.shuffle(IMAGE_IDS)
VAL_IMAGE_IDS = ['TCGA-G9-6363-01Z-00-DX1', 'TCGA-B0-5711-01Z-00-DX1']

############################################################
#  Dataset
############################################################

class MonusegDataset(utils.Dataset):

    def load_monuseg(self, dataset_dir, subset):
        """Load a subset of the monuseg dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nuclear", 1, "nuclear")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "nuclear",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    
    def read_data_and_mask_arr(self):
        print("Reading data array & mask array from disk")
        self.image_arr = np.array([self.load_image(x) for x in self._image_ids])
        
        # Since maks has different depth in each image, we store mask arr in dict
        self.mask_arr = {}
        for i in self._image_ids:
            self.mask_arr[i] = self.load_mask(i)[0]
        
    def load_mask_arr(self, mask_arr, image_id):
        """Get instance masks from image array
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        mask = self.mask_arr[image_id]
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclear":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
    

############################################################
#  Configurations
############################################################

# set checkpoint name
import datetime
name = str(datetime.datetime.now())
tmp = name.split(" ")
ckpt_name = tmp[0] + "_" + tmp[1][:8]
ckpt_name = ckpt_name.replace(":", "-")

# set training config and config NAME
config = monuconfig.Config()
config.NAME = config.NAME + "_" + ckpt_name

# set inference config
infer_config = monuconfig.InferConfig()
infer_config.NAME = ckpt_name
    
############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = MonusegDataset()
    dataset_train.load_monuseg(dataset_dir, subset)
    dataset_train.prepare()
    dataset_train.read_data_and_mask_arr()

    # Validation dataset
    dataset_val = MonusegDataset()
    dataset_val.load_monuseg(dataset_dir, "val")
    dataset_val.prepare()
    dataset_val.read_data_and_mask_arr()
    
    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    def image_channel_suffle(image):
        img_ = image.copy()
        r, g, b = img_[:,:,0], img_[:,:,1], img_[:,:,2]
        idx = [0, 1, 2]
        np.random.shuffle(idx)
        img_[:,:,idx[0]], img_[:,:,idx[1]], img_[:,:,idx[2]] = r, g, b 
        return img_

    def img_func(images, random_state, parents, hooks):
        for img in images:
            img = image_channel_suffle(img)
        return images

    def keypoint_func(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images
       
    augmentation = iaa.SomeOf((2, 5), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Affine(scale=(0.6, 2)),
        iaa.Lambda(img_func, keypoint_func),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 2.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=3*10e-5,
                epochs=1,
                augmentation=augmentation,
                layers='all')

def revise_predict_mask(mask, pred):
    mask = mask[12:1012, 12:1012, :]
    pred = pred[12:1012, 12:1012, :]

    prediction = np.zeros((1000,1000))
    for i in range(pred.shape[2]):
        prediction[pred[:,:,i]] = i + 1
        prediction = prediction.astype('uint16')
    return mask, prediction

# fast version of Aggregated Jaccrd Index
def agg_jc_index(mask, pred):

    """Calculate aggregated jaccard index for prediction & GT mask
    reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0
    mask: Ground truth mask, shape = [1000, 1000, instances]
    pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance
    Returns: Aggregated Jaccard index for GT & mask 
    """
    def compute_iou(m, pred, pred_mark_isused, idx_pred):
        # check the prediction has been used or not
        if pred_mark_isused[idx_pred]:
            intersect = 0
            union = np.count_nonzero(m)
        else:
            p = (pred == idx_pred)
            # replace multiply with bool operation
            s = ne.evaluate("m&p")
            intersect = np.count_nonzero(s)
            union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
        return (intersect, union)
    
    mask = mask.astype(np.bool)
    c = 0 # count intersection
    u = 0 # count union
    pred_instance = pred.max() # predcition instance number
    pred_mark_used = [] # mask used
    pred_mark_isused = np.zeros((pred_instance+1), dtype=bool)
    
    for idx_m in range(len(mask[0,0,:])):
        m = np.take(mask, idx_m, axis=2)     
        
        intersect_list, union_list = zip(*[compute_iou(m, pred, pred_mark_isused, idx_pred) for idx_pred in range(1, pred_instance+1)])

        iou_list = np.array(intersect_list) / np.array(union_list)    
        hit_idx = np.argmax(iou_list)
        c += intersect_list[hit_idx]
        u += union_list[hit_idx]
        pred_mark_used.append(hit_idx)
        pred_mark_isused[hit_idx+1] = True
        
    pred_mark_used = [x+1 for x in pred_mark_used]
    pred_fp = set(np.unique(pred)) - {0} - set(pred_mark_used)
    pred_fp_pixel = np.sum([np.sum(pred==i) for i in pred_fp])

    u += pred_fp_pixel
    return (c / u)

def evaluate(model, dataset_dir, model_path, config):
    print("EVALUATING...")
    dataset_val = MonusegDataset()
    dataset_val.load_monuseg(dataset_dir, "val")
    dataset_val.prepare()
    dataset_val.read_data_and_mask_arr()

    model.load_weights(model_path, by_name=True)

    ajx = []
    for i in dataset_val.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config, image_id=i, use_mini_mask=False, load_from_array=True)

        # Run object detection
        results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)
        r = results[0]

        # get resized mask & prediction from (1024, 1024) to (1000, 1000)
        y_true, y_pred = revise_predict_mask(gt_mask, r['masks'])

        # calculate the jc index
        aj_index = agg_jc_index(y_true, y_pred)
        ajx.append(aj_index)
    return np.array(ajx)

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')

    parser.add_argument('--dataset', required=False,
                        default="dataset/",
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        default="train",
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
#     parser.add_argument('--model_use', required=False,
#                         default="panet"
#                         metavar="Dataset sub-directory",
#                         help="Model structure used, choose 'panet' or 'maskrcnn' ")
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)
    
    # Check config setting
    config.display()
    
    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    T = time.time()
    train(model, args.dataset, args.subset)

    ## after training, run AJI evaluation
    model_infer = modellib.MaskRCNN(mode="inference", config=infer_config,
                                  model_dir=args.logs)

    trained_weights_path = model.checkpoint_path
    config.model_path = model.checkpoint_path
    print("LOADING WEIGHTS FROM {}".format(trained_weights_path))

    ajx_result = evaluate(model_infer, args.dataset, trained_weights_path, infer_config)
    print("MEAN AJI is {}".format(np.mean(ajx_result)))
    
    config.AJI_score = list(ajx_result)
    T2 = time.time()
    print("TOTAL TRAINING TIME: {} hours".format((T2-T)/3600))
    
    ## writing training parameter into google sheets


    path = 'miccai_gsheet.json'
    gc = pygsheets.authorize(service_file=path)
    sh = gc.open("miccai_maskrcnn")
    # get first sheet
    test = sh[0]
    # add columns
    test.add_cols(1)
    # get number of cols
    n_cols = test.cols

    # build config DataFrame
    config_list = []
    for a in dir(config):
        if not a.startswith("__") and not callable(getattr(config, a)):
            config_list.append([getattr(config, a)])

    df = pd.DataFrame(config_list)
    df[0] = df[0].apply(str)

    import string
    idx_dict = {}
    for i in range(1, 27):
        idx_dict[i] = string.ascii_uppercase[i-1]

    # append dataframe in new columns
    test.set_dataframe(df, "{}1".format(idx_dict[n_cols]))

    import pickle
    # save inference config for replication of performance
    with open('logs/{}/inference_config_{}.pickle'.format(config.NAME, config.NAME), 'wb') as f:
        pickle.dump(infer_config, f, -1)
