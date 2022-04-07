import numpy as np
import utils 

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
    """
    Compute the Histrogram of gradient by computing the magnitude and orientation 
    Then it stacks the magnitude of same orientation cells

    Args:
        gradient_dx : discrete derivation in the horizontal axis
        gradient_dy : discrete derivation in the vertical axis
        cell_size : tuple of the shape of the cells in the image
        resolution (int) : number of bins for the histogram
        multichannel (bool, optional): . Defaults to False.

    Returns:
        Histogram of orientation for each cell of the image
    """

    if multichannel:
        W, H, C = gradient_dx.shape
    else:
        W, H = gradient_dx.shape

    # Define the grid of cell 
    dw, dh = cell_size
    N = int(W // dw)
    M = int(H // dh)

    # Histrogram that we will iteratively compute
    orientation_histo = np.zeros((N, M, resolution))
    # step in angle between to bins of the histogram
    dtheta = 180/resolution
    
    if multichannel :
        # Compute the magnitude and orientation for all channel
        magnitudes_all = np.zeros(gradient_dx.shape)
        orientations_all = np.zeros(gradient_dx.shape)
        for c in range(C):
            magnitudes_all[:, :, c] = get_magnitude(gradient_dx[:, :, c], gradient_dy[:, :, c])
            orientations_all[:, :, c] = get_orientation(gradient_dx[:, :, c], gradient_dy[:, :, c])
        
        # Keep only the channel with higher magnetude
        keep_c = np.argmax(magnitudes_all, axis = -1)
        # Need a meshgrid in order to retrieve the argmax correctly on the array
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
                # Compute the value for each cell
                orientation_cell = orientations[n*dw :(n+1)*dw, m*dh: (m +1)*dh]
                magnitude_cell = magnitudes[n*dw :(n+1)*dw, m*dh: (m +1)*dh]
                # Call the computation of the histogram for one cell
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
    """
    Get al the block of the images and call the function that normalize each blocks
    """
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
    """
    Compute the Histogram of Oriented Gradient 
    Inspired by Sckimage code that help us to deeply understand the method and its specificity
    """
    # Important to change the type since we compute gradient
    image = image.astype(float)

    # Need to deal with mulltichannel images
    if multichannel:
        gradient_dx, gradient_dy = compute_gradient_multi_channel(image)
    else:
        gradient_dx, gradient_dy = compute_gradient(image)
    
    # Call the computation of the features then the normalization function
    hog = histogram_gradient(gradient_dx, gradient_dy, cell_size, resolution, multichannel)
    hog_normalize = normalizing_blocks(hog, block_size, method)

    # We deal with images so 2D information but we may want to flatten it for our classifier 
    if flatten:
        return hog_normalize.flatten()
    return hog_normalize
