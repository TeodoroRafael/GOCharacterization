#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:01:01 2021

@author: rafaelteodoro
"""


import cv2             #!pip install opencv-python-headless
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import pandas as pd


def measurament(image, proportion_pix_um = 4.34):

    #STEP1 - Read image, define image size, and convert pixels
    mask = (image - 1) * -1

    mask_shape = '{}'.format(mask.shape)
    
    pixels_to_um = proportion_pix_um # 1 pixel = 4.34 um or 1 pixel = 0.23 um, etc.

    #STEP 2: Label flakes in the masked image
    s = [[1,1,1],[1,1,1],[1,1,1]]
    labeled_mask, num_labels = ndimage.label(mask, structure=s)
    
    img_color = color.label2rgb(labeled_mask, bg_label=0)
    
    #Step 5: Measure the properties of each flake
    clusters = measure.regionprops(labeled_mask, image)  #send in original image for Intensity measurements
        
    #Step 6: Output results into a csv file   
        
    propList = ['centroid',
                'Area',
                'equivalent_diameter',
                'orientation', 
                'MajorAxisLength',
                'MinorAxisLength',
                'Perimeter']    
        
    
    output_file = open('image_measurements.csv', 'w')
    output_file.write(',' + ",".join(propList) + '\n') 
    
    for cluster_props in clusters:
        #output cluster properties to the excel file
        output_file.write(str(cluster_props['Label']))
        for i,prop in enumerate(propList):
            if(prop == 'Area'): 
                to_print = cluster_props[prop]*(pixels_to_um**2)   #Convert pixel square to um square
            elif(prop == 'orientation'): 
                to_print = cluster_props[prop]*57.2958  #Convert to degrees from radians
            else: 
                to_print = cluster_props[prop]     #Reamining props, basically the ones with Intensity in its name
            output_file.write(',' + str(to_print))
        output_file.write('\n')
    output_file.close()   #Closes the file, otherwise it would be read only. 
    
    #Importing DataFrame and preparing Data
    df = pd.DataFrame(pd.read_csv('/content/image_measurements.csv', sep=','))

    #Fixing data location label type
    for i_text in range(0,len(df.iloc[:,0])):
        df.iloc[i_text,0] = int(float(df.iloc[i_text, 0].replace('(','')))
        df.iloc[i_text,1] = int(float(df.iloc[i_text, 1].replace(')','')))
    
    #Creating tuples for location
    loc_label = list(zip(df.loc[:]['centroid'],df.loc[:]['Unnamed: 0']))
    df['centroid'] = loc_label
    
    #Deleting first
    df = df.drop(columns=['Unnamed: 0'], axis=0)
    
    df.to_csv(r'./image_measurements.csv')
    return img_color, df

 