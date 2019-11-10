import numpy as np
import matplotlib.pyplot as plt
from cv2 import imwrite
from pdb import set_trace
from os import makedirs
from random import shuffle
from os.path import join, splitext, basename, isdir
from sys import stdout, exit
from shutil import copyfile
from skimage.draw import polygon
from json import load
from argparse import ArgumentParser
from glob import glob

def make_label_mask(json_obj, jpg_file):

    with open(json_obj, 'r') as f:
        polygon_obj = load(f)['labels']

    im = plt.imread(jpg_file) # assumes jpg is (widthxheightxchannels)
    labels = np.zeros(im.shape[:2]) # grayscale 
    for label in polygon_obj:
        row_coords = [p['y'] for p in label['vertices']]
        col_coords = [p['x'] for p in label['vertices']]
        rr, cc = polygon(row_coords, col_coords)
        labels[rr, cc] = 255
    return labels

def label_masks(jpgs, jsons, out_directory):
    out_dir_mask = join(out_directory, 'masks')
    out_dir_image = join(out_directory, 'images')
    
    if not isdir(out_dir_mask):
        makedirs(out_dir_mask)
    if not isdir(out_dir_image):
        makedirs(out_dir_image)
    
    for i, (jpg, json) in enumerate(zip(jpgs, jsons)):
        try:
            labels = make_label_mask(json, jpg)
            copyfile(jpg, join(out_dir_image, basename(jpg)))
            stdout.write("{}/{}\r".format(i+1, len(jsons)))
            imwrite(join(out_dir_mask, splitext(basename(jpg))[0] + '.png'), labels)
        except UnicodeDecodeError as e:
            print("json file {} may be corrupted".format(json))
    stdout.write("\n")
        
        

def _filter_jsons_and_jpgs(jsons, jpgs):
    out_jpgs = []
    out_jsons = []
    for jpg in jpgs:
        matching_json = None
        for json in jsons:
            if splitext(jpg)[0] in json:
                out_jsons.append(json)
                out_jpgs.append(jpg)
    assert(len(out_jpgs) == len(out_jsons))
    return out_jpgs, out_jsons



if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('--json-jpg-dir', type=str, required=True)
    ap.add_argument('--train-dir', type=str, required=True)
    ap.add_argument('--test-dir', type=str, required=True)
    ap.add_argument('--split', type=float, default=0.85)

    args = ap.parse_args()

    jsons = sorted(glob(join(args.json_jpg_dir, '*json')))
    jpgs = sorted(glob(join(args.json_jpg_dir, '*jpg')))
    jpgs, jsons = _filter_jsons_and_jpgs(jsons, jpgs)
    jsons = np.asarray(jsons)
    jpgs = np.asarray(jpgs)
    # do train test split here
    indices = np.arange(len(jsons))
    n_train = int(len(jsons) * args.split)
    n_test = len(jsons) - n_train
    train_indices = np.random.choice(indices, n_train, replace=False)
    test_indices = list(set(indices) - set(train_indices)) # very python3?
    train_jpgs = jpgs[train_indices]
    train_jsons = jsons[train_indices]
    test_jpgs = jpgs[test_indices]
    test_jsons = jsons[test_indices]
    label_masks(train_jpgs, train_jsons, args.train_dir)
    label_masks(test_jpgs, test_jsons, args.test_dir)

