import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use same padding (mode = 'reflect'). Refer docs for further info.

from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space)


def gaussian_filter(image, sigma):
    """
    Given an image, apply a Gaussian filter with the input kernel size
    and standard deviation

    Input
      image: image of size HxW
      sigma: scalar standard deviation of Gaussian Kernel

    Output
      Gaussian filtered image of size HxW
    """
    H, W = image.shape
    # -- good heuristic way of setting kernel size
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    # Ensure that the kernel size isn't too big and is odd
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    # TODO implement gaussian filtering of size kernel_size x kernel_size
    # Similar to Corner detection, use scipy's convolution function.
    # Again, be consistent with the settings (mode = 'reflect').
    
    # create gaussian kernel
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    
    output = scipy.ndimage.convolve(image, kernel, mode='reflect')
    return output




def main():
    image = read_img('polka.png')
    # import pdb; pdb.set_trace()
    # Create directory for polka_detections
    if not os.path.exists("./polka_detections"):
        os.makedirs("./polka_detections")

    # -- TODO Task 8: Single-scale Blob Detection --

    # (a), (b): Detecting Polka Dots
    # First, complete gaussian_filter()
    print("Detecting small polka dots")
    # -- Detect Small Circles
    k = 1.5
    sigma_1 = 3.3
    sigma_2 = k * sigma_1
    gauss_1 = gaussian_filter(image, sigma_1)  # to implement
    gauss_2 = gaussian_filter(image, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_small = gauss_2 - gauss_1  # to implement

    # visualize maxima
    maxima = find_maxima(DoG_small, k_xy=10)
    visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_small.png')
    plt.clf() 
    
    
    # -- Detect Large Circles
    print("Detecting large polka dots")
    k = 1.5
    sigma_1 = 7
    sigma_2 = k * sigma_1
    gauss_1 = gaussian_filter(image, sigma_1)  # to implement
    gauss_2 = gaussian_filter(image, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_large = gauss_2 - gauss_1  # to implement

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_large.png')
    plt.clf() 

    # # # -- TODO Task 9: Cell Counting --
    print("Detecting cells")

    cell_1 = read_img("./cells/008cell.png")
    cell_2 = read_img("./cells/004cell.png")
    cell_3 = read_img("./cells/005cell.png")
    cell_4 = read_img("./cells/006cell.png")
    cells = [cell_1, cell_2, cell_3, cell_4]
    
    # Detect the cells in any four (or more) images from vgg_cells
    # Create directory for cell_detections
    if not os.path.exists("./cell_detections"):
        os.makedirs("./cell_detections")

    
    print("Detecting cell1")
    k = 3
    sigma_1 = 3.7
    sigma_2 = k * sigma_1
    gauss_1 = gaussian_filter(cell_1, sigma_1)  # to implement
    gauss_2 = gaussian_filter(cell_1, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_cell1 = gauss_2 - gauss_1  # to implement

    # visualize maxima
    maxima = find_maxima(DoG_cell1, k_xy=10)
    visualize_scale_space(DoG_cell1, sigma_1, sigma_2 / sigma_1,
                        './cell_detections/cell1_DoG.png')
    visualize_maxima(cell_1, maxima, sigma_1, sigma_2 / sigma_1,
                    './cell_detections/cell1.png')
    plt.clf() 
    
    
    print("Detecting cell2")
    k = 3
    sigma_1 = 3.4
    sigma_2 = k * sigma_1
    gauss_1 = gaussian_filter(cell_2, sigma_1)  # to implement
    gauss_2 = gaussian_filter(cell_2, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_cell2 = gauss_2 - gauss_1  # to implement

    # visualize maxima
    maxima = find_maxima(DoG_cell2, k_xy=10)
    visualize_scale_space(DoG_cell2, sigma_1, sigma_2 / sigma_1,
                        './cell_detections/cell2_DoG.png')
    visualize_maxima(cell_2, maxima, sigma_1, sigma_2 / sigma_1,
                    './cell_detections/cell2.png')
    plt.clf() 
    
    
    
    
    print("Detecting cell3")
    k = 5
    sigma_1 = 3.7
    sigma_2 = k * sigma_1
    gauss_1 = gaussian_filter(cell_3, sigma_1)  # to implement
    gauss_2 = gaussian_filter(cell_3, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_cell3 = gauss_2 - gauss_1  # to implement

    # visualize maxima
    maxima = find_maxima(DoG_cell3, k_xy=10)
    visualize_scale_space(DoG_cell3, sigma_1, sigma_2 / sigma_1,
                        './cell_detections/cell3_DoG.png')
    visualize_maxima(cell_3, maxima, sigma_1, sigma_2 / sigma_1,
                    './cell_detections/cell3.png')
    plt.clf() 
    
    
    
    
    print("Detecting cell4")
    k = 1.9
    sigma_1 = 3.7
    sigma_2 = k * sigma_1
    gauss_1 = gaussian_filter(cell_4, sigma_1)  # to implement
    gauss_2 = gaussian_filter(cell_4, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_cell4 = gauss_2 - gauss_1  # to implement

    # visualize maxima
    maxima = find_maxima(DoG_cell4, k_xy=10)
    visualize_scale_space(DoG_cell4, sigma_1, sigma_2 / sigma_1,
                        './cell_detections/cell4_DoG.png')
    visualize_maxima(cell_4, maxima, sigma_1, sigma_2 / sigma_1,
                    './cell_detections/cell4.png')
    plt.clf() 

    

if __name__ == '__main__':
    main()
