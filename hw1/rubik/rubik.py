import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import argparse
import os

if __name__ == "__main__" :
    print(f"Runing $ python rubik.py <source> <target> <cvttolab> \nPlease give the input. For <cvttolab>, 0 for use RGB, 1 for use LAB")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('cvttolab', type=int, help="0 for RGB, 1 for LAB")
    args = parser.parse_args()
    
    if not os.path.exists(args.target):
        os.mkdir(args.target)
    
    # get image names
    images = [fn for fn in os.listdir(args.source) if fn.endswith(".png")]
    images.sort()

    # process images
    result = {}
    for image in images:
        if image not in result:
            result[image] = []
        
        I = cv2.imread(os.path.join(args.source, image))
        
        # RGB
        if not args.cvttolab:
            for i in range(I.shape[2]):
                result[image].append(I[:, :, i]) 
        # LAB
        else:
            I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
            for i in range(I_LAB.shape[2]):
                result[image].append(I_LAB[:, :, i]) 
    
    # show images 
    color = 'RGB'
    if args.cvttolab:
        color = 'LAB'
        
    for image in result:
        [plt.imsave(f'result/{image}_{color}_{i}.png',image_channel, cmap='gray') for i, image_channel in enumerate(result[image])]
        
    print(f'Image saved!')
    
    