import numpy as np
from os import path
import json
from skimage.io import imread

datadir  = 'attributes_dataset/train/'

with open(path.join(datadir, 'labels.txt'), 'r') as f:
    rows = f.readlines()

data = {}
for i, row in enumerate(rows):
    fields = [field.strip() for field in row.split()]
    image_name = fields[0]
    bbox = []
    if fields[1] == 'NaN':
        img = imread(datadir + image_name)
        bbox = [0, 0, img.shape[1], img.shape[0]]
    else:
        bbox = list(np.array(fields[1:5]).astype(float))
    data[image_name] = [{'x':bbox[0], 'y':bbox[1], 'w':bbox[2], 'h':bbox[3]}]

with open(path.join(datadir, 'annotation.json'), 'w') as f:
    json.dump(data, f, indent=2)



