import numpy as np
import utils 

import pandas as pd
from matplotlib import pyplot as plt


def compute_gradient(image):
    """ 
    Compute the arrays of gradient in dx and dy
    We have dx(t) = x(t + 1) - x(t - 1)
    Suppose image is a W x H array 
    """
    gradient_dx = np.zeros(image.shape)
    gradient_dy = np.zeros(image.shape)
    gradient_dx[1:-1, :] = image[2:, :] - image[:-2, :]
    gradient_dy[:, 1:-1] = image[:, 2:] - image[:, :-2]
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
        gradient_dx[:, :, c], gradient_dy[:, :, c]  = compute_gradient(image[:, :, c])

    return gradient_dx, gradient_dy

def get_orientation(gradient_dx, gradient_dy):
    """Get the orientation between arrays"""

    angles_rad = np.arctan2(gradient_dx, gradient_dy)
    angles_deg = angles_rad * 180/ np.pi
    angles_deg = angles_deg % 180
    return angles_deg

def get_magnitude(gradient_dx, gradient_dy):
    """Get the magnitude of the gradient"""
    return np.sqrt(np.power(gradient_dx, 2) + np.power(gradient_dy, 2))

def histogram_gradient_cell(orientation, magnitude, theta_beg, theta_end):
    """ Get mean magnetude of the pixel in the range """
    W, H = orientation.shape

    pixel_concerned = (orientation >= theta_beg) & (orientation < theta_end)
    magnitude_concerned = magnitude * pixel_concerned
    sum_magnetude = np.sum(magnitude_concerned)

    return sum_magnetude/(W * H)

def histogram_gradient(gradient_dx, gradient_dy, cell_size, resolution, multichannel = False):

    if multichannel:
        W, H, C = gradient_dx.shape
    else:
        W, H = gradient_dx.shape
    dw, dh = cell_size
    N = int(W // dw)
    M = int(H // dh)

    orientation_histo = np.zeros((N, M, resolution))
    dtheta = 180/resolution
    
    if multichannel :
        magnitudes_all = np.zeros(gradient_dx.shape)
        orientations_all = np.zeros(gradient_dx.shape)
        for c in range(C):
            magnitudes_all[:, :, c] = get_magnitude(gradient_dx[:, :, c], gradient_dy[:, :, c])
            orientations_all[:, :, c] = get_orientation(gradient_dx[:, :, c], gradient_dy[:, :, c])
        
        keep_c = np.argmax(magnitudes_all, axis = -1)
        rows, cols = np.meshgrid(np.arange(W),
                             np.arange(H),
                             indexing='ij',
                             sparse=True)
        magnitudes = magnitudes_all[rows, cols, keep_c]
        orientations = orientations_all[rows, cols, keep_c]
    else:
        orientations = get_orientation(gradient_dx, gradient_dy)
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

def normalization(block, method, eps = 1e-5):
    """
    Do the normalization of the block according to the chosen method
    Espilon used for avoiding divison by 0
    """
    if method == 'L1':
        block_normalized = block / (np.sum(np.abs(block)) + eps)

    elif method == 'L2':
        block_normalized = block / np.sqrt((np.sum(np.power(block, 2)) + eps))
    
    else:
        block_normalized = block
    return block_normalized

def normalizing_blocks(hog, block_size, method):
    N, M, O = hog.shape
    brows, bcols = block_size

    # Compute the new shapes of the array : do slidding normalization
    block_rows = N - brows + 1
    block_cols = M - bcols + 1

    hog_normalized = np.zeros((block_rows, block_cols, brows, bcols, O ))

    for r in range(block_rows):
        for c in range(block_cols):
            block = hog[r: r + brows, c: c + bcols, :]
            hog_normalized[r, c, :] =  normalization(block, method)
    return hog_normalized

def Histogram_oriented_gradient(image, resolution = 9, cell_size = (4, 4), block_size= (4, 4), multichannel = False, method = 'L1', flatten = True, visualize = True):
    image = image.astype(float)
    if multichannel:
        gradient_dx, gradient_dy = compute_gradient_multi_channel(image)

    else:
        gradient_dx, gradient_dy = compute_gradient(image)
    
    hog = histogram_gradient(gradient_dx, gradient_dy, cell_size, resolution, multichannel)
    hog_normalize = normalizing_blocks(hog, block_size, method)
    if flatten:
        return hog_normalize.flatten()
    return hog_normalize



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from skimage.feature import hog
    from skimage import data, exposure


    image = data.astronaut()
    vis = True
    fd, hog_image = Histogram_oriented_gradient(image, resolution = 9, cell_size = (8, 8), block_size= (4, 4), multichannel = True, flatten = False, visualize = vis)
    #fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),  cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    if vis :
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Input image')

        plt.show()

        
