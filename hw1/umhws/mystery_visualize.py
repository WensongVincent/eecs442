#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(name, X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    vmax = np.nanmax(X)
    vmin = np.nanmin(X)
    N = colors.shape[0]
    X = ((N - 1) * (X - vmin) / (vmax - vmin)).astype(np.uint8)
    data_return = np.zeros((X.shape[0], X.shape[1], 3))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            data_return[i, j, :] = colors[X[i, j]]
    
    plt.imsave(name, np.log1p(data_return))
    data_return = ((N - 1) * data_return).astype(np.uint8)
    return data_return


if __name__ == "__main__":
    # Example
    colors = np.load("mysterydata/colors.npy")
    data1 = np.load("mysterydata/mysterydata.npy")
    dataNone = [plt.imsave(f"vis_1_{i}.png", data1[:, :, i]) for i in range(9)]
    
    # Question 1
    data2 = np.load("mysterydata/mysterydata2.npy")
    data2 = np.log1p(data2)
    dataNone = [plt.imsave(f"vis_2_{i}.png", data2[:, :, i]) for i in range(9)]
    
    # Question 2
    data3 = np.load("mysterydata/mysterydata3.npy")
    # print(np.mean(np.isfinite(data3)))
    # print(np.mean(np.isnan(data3)))
    dataNone = [plt.imsave(f"vis_3_{i}.png", data3[:, :, i], vmin = np.nanmin(data3[:, :, i]), vmax = np.nanmax(data3[:, :, i])) for i in range(9)]

    #Question 3 and 4
    data4 = np.load("mysterydata/mysterydata4.npy")
    data4_colormap = [colormapArray(f"vis_4_{i}.jpg", data4[:, :, i], colors) for i in range(9)]
    
    
    