"""
Homography fitting functions
You should write these
"""
import numpy as np
from common import homography_transform

def fit_homography(XY):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit a homography from [x,y,1] to [x',y',1].
    
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
    Output -H: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

    '''
    N = XY.shape[0]
    A = np.zeros((2*N, 9))
    for i in range(N):
        x, y, xp, yp = XY[i]
        # A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        # A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        A[2*i] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        A[2*i+1] = [x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp]

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(A)
    
    # The solution is the last column of V (or the last row of V transpose)
    h = Vt[-1]
    # Normalize h
    h /= np.linalg.norm(h)
    # Reshape h to get the homography matrix H
    H = h.reshape(3, 3)
    H /= H[-1, -1]
    return H


def RANSAC_fit_homography(XY, eps=1, nIters=1000):
    '''
    Perform RANSAC to find the homography transformation 
    matrix which has the most inliers
        
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            eps: threshold distance for inlier calculation
            nIters: number of iteration for running RANSAC
    Output - bestH: a (3,3) homography matrix fit to the 
                    inliers from the best model.

    Hints:
    a) Sample without replacement. Otherwise you risk picking a set of points
       that have a duplicate.
    b) *Re-fit* the homography after you have found the best inliers
    '''
#     bestH, bestCount, bestInliers = np.eye(3), -1, np.zeros((XY.shape[0],))
#     bestRefit = np.eye(3)

    # Initialize the best homography matrix, inlier count and inlier set
    bestH = None
    bestCount = -1
    bestInliers = None

    for _ in range(nIters):
        # Step 1: Randomly select 4 pairs of points without replacement
        indices = np.random.choice(XY.shape[0], 4, replace=False)
        sample = XY[indices]

        # Step 2: Compute the homography matrix using the provided utility function
        H = fit_homography(sample)

        # Step 3: Apply homography and determine inliers
        # Transform source points to destination plane
        homogenized_src_pts = np.concatenate((XY[:, :2], np.ones((XY.shape[0], 1))), axis=1)
        transformed_pts = np.dot(H, homogenized_src_pts.T).T
        transformed_pts /= transformed_pts[:, 2][:, np.newaxis]  # Normalize

        # Calculate distances from actual to projected points
        homogenized_dst_pts = np.concatenate((XY[:, 2:], np.ones((XY.shape[0], 1))), axis=1)
        distances = np.linalg.norm(homogenized_dst_pts[:, :2] - transformed_pts[:, :2], axis=1)

        # Inliers are points with distance less than epsilon
        inliers = distances < eps
        inlier_count = np.sum(inliers)

        # Step 4: Keep track of the best homography with the most inliers
        if inlier_count > bestCount:
            bestCount = inlier_count
            bestH = H
            bestInliers = inliers

    # Step 5: Re-fit the homography using all inliers from the best model found
    if bestInliers is not None and bestCount > 4:  # More than the minimal sample size
        all_inliers = XY[bestInliers]
        bestH = fit_homography(all_inliers)
    else:
        bestH = np.eye(3)  # Fallback to identity matrix if no good model is found

    return bestH

if __name__ == "__main__":
    #If you want to test your homography, you may want write any code here, safely
    #enclosed by a if __name__ == "__main__": . This will ensure that if you import
    #the code, you don't run your test code too
    pass
