#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 12:49:07 2025

@author: yunusa2k2
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CodedImagesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.image_serials = []

        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for image_filename in os.listdir(subfolder_path):
                    if image_filename.endswith(".jpg"):
                        image_serial = image_filename.split("Image")[1].split(".jpg")[0]
                        self.image_paths.append(os.path.join(subfolder_path, image_filename))
                        self.image_serials.append(int(image_serial))

        self.serial_to_timeperiod = dict(zip(self.data_info['Q3ImageNumber'], self.data_info['Q2TimePeriod']))
        self.serial_to_relevancy = dict(zip(self.data_info['Q3ImageNumber'], self.data_info['Q4Relevancy']))
        self.serial_to_urgency = dict(zip(self.data_info['Q3ImageNumber'], self.data_info['Q5Urgency']))
        self.serial_to_damage = dict(zip(self.data_info['Q3ImageNumber'], self.data_info['ContainsMotif3']))
        self.serial_to_relief = dict(zip(self.data_info['Q3ImageNumber'], self.data_info['ContainsMotif10']))        
        self.serial_to_text = dict(zip(self.data_info['Q3ImageNumber'], self.data_info['Q9Imageattributes']))
        
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_serial = self.image_serials[idx]

        # Get labels
        time_period = self.serial_to_timeperiod.get(image_serial, 0)
        relevancy = self.serial_to_relevancy.get(image_serial, 0)
        urgency = self.serial_to_urgency.get(image_serial, 0)
        damage = self.serial_to_damage.get(image_serial, 0)
        relief = self.serial_to_relief.get(image_serial, 0)
        text = self.serial_to_text.get(image_serial, "")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, image_serial, time_period, relevancy, urgency, damage, relief, text
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    