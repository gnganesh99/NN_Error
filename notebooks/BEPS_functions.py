import numpy as np
import matplotlib.pyplot as plt
import math
import os

from sklearn.model_selection import train_test_split
from atomai.utils import get_coord_grid, extract_patches_and_spectra, extract_subimages


def BEPS_image_spectral_pairs(beps_file_path, window_size = 16, step = 1):


    input_file = np.load(beps_file_path)
    
    image = input_file['image']
    #print(image.shape)
    spectra = input_file['spectra']
    #print(spectra.shape)
    v_step = input_file['spec_step_vol']

    
    
    # Extract patches
    coordinates = get_coord_grid(image, step = step, return_dict=False)

    # extract image patch for each point on a grid
    window_size = window_size
    features_all, coords, _ = extract_subimages(image, coordinates, window_size)
    patches = features_all[:,:,:,0]

    indices_all = np.array(coords, dtype = int)
    
    # extract spectra 
    n = patches.shape[0]
    all_spectra = []

    for ind in range(n):
        spectrum =  spectra[indices_all[ind,0], indices_all[ind,1]]  # indices convention is reversed for the spectra
        all_spectra.append(spectrum)

    all_spectra = np.array(all_spectra)

    return patches, all_spectra, indices_all, v_step


def extract_beps_data(beps_file_path):
    input_file = np.load(beps_file_path)
    
    image = input_file['image']
    #print(image.shape)
    spectra = input_file['spectra']
    #print(spectra.shape)
    v_step = input_file['spec_step_vol']

    return image, spectra, v_step

