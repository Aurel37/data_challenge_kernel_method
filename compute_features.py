import numpy as np
import utils 

import pandas as pd
from matplotlib import pyplot as plt


def compute_gradient(image):
    """ 
    Compute the arrays of gradient in dx and dy
    Suppose image is a W x H array 
    """

    gradient_dx = np.zeros(image.shape)
    gradient_dy = np.zeros(image.shape)
    gradient_dx[:-1, :] = image[1:, :] - image[:-1, :]
    gradient_dy[:, :-1] = image[:, 1:] - image[:, :-1]

    return gradient_dx, gradient_dy


def compute_gradient_multi_channel(image):
    """ 
    Compute the arrays of gradient in dx and dy
    for each channel
    Suppose image is a W x H x C array 
    """
    W, H, C = image.shape
    gradient_dx = np.zeros((W, H, C))
    gradient_dy = np.zeros((W, H, C))
    for c in range(C):
        gradient_dx[:, :, c], gradient_dy[:, :, c]  = commpute_gradient(image[:, :, c])

    return gradient_dx, gradient_dy

def get_oriantation(gradient_dx, gradient_dy):
    """Get the orientation between arrays"""
    angles_rad = np.arctan2(gradient_dx, gradient_dy)
    angles_deg = angles_rad * 180/ np.pi
    angles_deg %= 180

    return angles_deg

def get_magnitude(gradient_dx, gradient_dy):
    """Get the magnitude of the gradient"""
    return np.sqrt(np.power(gradient_dx, 2) + np.power(gradient_dy, 2))

def histogram_gradient_cell(orientation, magnitude, theta_beg, theta_end):
    """ Get mean magnetude of the pixel in the range """
    rows, cols = orientation.shape
    pixel_concerned = (orientation >= theta_beg) & (orientation < theta_end)
    sum_magnetude = np.sum(magnitude[pixel_concerned])
    return sum_magnetude/(rows * cols)

def histogram_gradient(gradient_dx, gradient_dy, cell_size, resolution):

    W, H = gradient_dx.shape
    dw, dh = cell_size
    N = int(W // dw)
    M = int(H // dh)

    orientation_histo = np.zeros((N, M, resolution))
    dtheta = 180/resolution

    orientations = get_oriantation(gradient_dx, gradient_dy)
    magnitudes = get_magnitude(gradient_dx, gradient_dy)

    for t in range(resolution):
        theta_beg = dtheta*t
        theta_end = dtheta*(t + 1)
        for n in range(N):
            for m in range(M):
                orientation_cell = orientations[n*dw :(n+1)*dw, m*dh: (m +1)*dh]
                magnitude_cell = magnitudes[n*dw :(n+1)*dw, m*dh: (m +1)*dh]
                orientation_histo[n, m, t] = histogram_gradient_cell(orientation_cell, magnitude_cell, theta_beg, theta_end)

    return orientation_histo


def Histogram_oriented_gradient(image, resolution = 9, cell_size = (4, 4), multichannel = False,):

    if multichannel:
        gradient_dx, gradient_dy = compute_gradient_multi_channel(image)

    else:
        gradient_dx, gradient_dy = compute_gradient(image)


    hog = histogram_gradient(gradient_dx, gradient_dy, cell_size, resolution)
    
    return hog