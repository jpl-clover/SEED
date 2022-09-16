from __future__ import print_function
import io
import csv
import tqdm
import torch
import pickle
import base64
import random
import numpy as np
import PIL
from PIL import Image
from tools.tsv_io import TSVFile
from torch.utils.data import Dataset


class TSVDataset(Dataset):
    """ TSV dataset for ImageNet 1K training
    """    
    def __init__(self, tsv_file, transform=None, target_transform=None):
        self.tsv = TSVFile(tsv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        row = self.tsv.seek(index)
        image_data = base64.b64decode(row[-1])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        target = int(row[1])

        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return round(self.tsv.num_rows())


class Small_Patch_TSVDataset_Legacy(Dataset):
    """
        TSV dataset for ImageNet 1K training with jigsaw random crop.
    """
    def __init__(self, tsv_file, transform=None, jigsaw_transform=None, num_patches=6):
        self.tsv = TSVFile(tsv_file)
        self.transform = transform
        self.jigsaw_transform = jigsaw_transform
        self.num_patches = num_patches

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        row = self.tsv.seek(index)
        image_data = base64.b64decode(row[-1])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        img = self.transform(image)

        # Stack small images
        small_patches = []
        for img_index in range(self.num_patches):
            small_patches.extend(
                self.jigsaw_transform(image).unsqueeze(0))

        return img, torch.stack(small_patches)

    def __len__(self):
        return round(self.tsv.num_rows())

class DatasetFromFilenamesAndLabels(Dataset):
    def __init__(
            self, 
            fp_images, 
            fp_filenames_and_labels,
            fp_classes,
            verbose=True
        ):

        """ 
        Creates a dataset defined by a file of filenames and labels
        """
        
        def parse_filename_and_label_file(fp):
            # Helper to parse split file data into list of tuples
            with open(fp, "r") as f:
                content = [l.strip().split() for l in f.readlines()]
            xs = np.array([c[0] for c in content])
            ys =  np.array([int(c[1]) for c in content])
            return xs, ys
        
        # Where are the images saved?
        self.fp_images = fp_images
        
        # Parse the filename and label file
        self.filenames, self.labels = parse_filename_and_label_file(fp_filenames_and_labels)
        
        # Load the class map
        self.classes = np.loadtxt(fp_classes, delimiter=',', usecols=(-1), dtype="str")
        
        # Provide some information
        if verbose:
            self._print_stats()

    def _print_stats(self):

        # Print out so we can check if we got the right amount of each class 
        # as desired (with over and under sampling)
                    
        values, counts = np.unique(self.labels, return_counts=True)

        print("Label indices / counts:")
        for i, v in enumerate(values):
            print(f"\t{v} : {counts[i]}")

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """ 
        An instance in this dataset is just a labelled image. This returns that image with the
        desired transforms + the label
        """
        
        # Load image, and force three channels
        try:
            image = PIL.Image.open(f"{self.fp_images}/{self.filenames[idx]}").convert("RGB")
        except:
            image = np.zeros((224, 224, 3), dtype=np.float32)
            
        return image, self.labels[idx]

class Small_Patch_TSVDataset(Dataset):
    """
    New dataset file for MSL data
    """
    def __init__(self, tsv_file, transform=None, jigsaw_transform=None, num_patches=6):
        # Load the train, test, validation mapping file
        fp_msl = "/home/08328/isaacrw/clover_shared/datasets/msl-labeled-data-set-v2.1/"
        
        # Use the above class as a helper
        split = "train"
        self.dataset = DatasetFromFilenamesAndLabels(
            f"{fp_msl}/images/",
            f"{fp_msl}/{split}-set-v2.1.txt",
            f"{fp_msl}/class_map.csv",
            verbose=True,
        )

        #self.tsv = TSVFile(tsv_file)
        self.transform = transform
        self.jigsaw_transform = jigsaw_transform
        self.num_patches = num_patches

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        # Load the image from the MSL dataset
        image = self.dataset[index][0]

        #row = self.tsv.seek(index)
        #image_data = base64.b64decode(row[-1])
        #image = Image.open(io.BytesIO(image_data))
        #image = image.convert('RGB')
        img = self.transform(image)

        # Stack small images
        small_patches = []
        for img_index in range(self.num_patches):
            small_patches.extend(
                self.jigsaw_transform(image).unsqueeze(0))

        return img, torch.stack(small_patches)

    def __len__(self):
        #return round(self.tsv.num_rows())
        return len(self.dataset)

