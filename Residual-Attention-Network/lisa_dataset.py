from cv2 import imread
from os import path
from torch.utils.data import Dataset as DatasetBase
from torchvision import transforms

def readImage(img_path):
    image = imread(img_path) # <class 'numpy.ndarray'>
    # Alternative way
    #from PIL import Image
    #image = Image.open(img_path) <class 'PIL.PngImagePlugin.PngImageFile'>
    #import numpy as np
    #image = np.array(image) # <class 'numpy.ndarray'>
    return image

def make_catagories(filePath):
    catagories = {}
    with open(filePath) as f:
        allLines = f.readlines()
        for line in allLines:
            splited = line.strip().split(',')
            catagories[splited[0]] = int(splited[1])
    return catagories

def make_dataset(categories, filePath):
    allAnnoLines = []
    with open(filePath) as f:
        f.readline() # Discard the header line.
        lines = f.readlines()
        for line in lines:
            splited = line.strip().split(',')
            if splited[1] in categories: # Discard the line out of catagories
                allAnnoLines.append(line)
    return allAnnoLines

class LisaDataset(DatasetBase):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.categories = make_catagories(path.join(root, 'categories.csv'))
        self.datasetAnnoLines = make_dataset(self.categories, path.join(root, 'annotations.csv'))

    def __getitem__(self, idx):
        line = self.datasetAnnoLines[idx]
        splited = line.strip().split(',')
        image = readImage(path.join(self.root, splited[0]))
        image = self.transform(image) # To torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        label = self.categories[splited[1]]
        return image, label
    
    def __len__(self):
        return len(self.datasetAnnoLines)
