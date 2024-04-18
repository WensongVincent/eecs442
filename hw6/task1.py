from utils import dehomogenize, homogenize, draw_epipolar
import numpy as np
import cv2
import pdb
import os


def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """

    #This will give you an answer you can compare with
    #Your answer should match closely once you've divided by the last entry
    FOpenCV, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

    F = np.eye(3)
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    # Normalize the points to increase accuracy
    pts1_hom = homogenize(pts1)
    pts2_hom = homogenize(pts2)

    # Center and scale points for numerical stability
    mean1 = np.mean(pts1, axis=0)
    mean2 = np.mean(pts2, axis=0)
    std1 = np.std(pts1)
    std2 = np.std(pts2)

    # Transformation matrices for normalization
    T1 = np.array([
        [1/std1, 0, -mean1[0]/std1],
        [0, 1/std1, -mean1[1]/std1],
        [0, 0, 1]
    ])
    T2 = np.array([
        [1/std2, 0, -mean2[0]/std2],
        [0, 1/std2, -mean2[1]/std2],
        [0, 0, 1]
    ])

    # Normalize points
    pts1_norm = (T1 @ pts1_hom.T).T
    pts2_norm = (T2 @ pts2_hom.T).T

    # Create matrix A for the linear equation system Ax = 0
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        x1, y1, _ = pts1_norm[i]
        x2, y2, _ = pts2_norm[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # Solve the homogeneous equation system using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce the rank constraint (rank 2)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Set smallest singular value to 0
    F = U @ np.diag(S) @ Vt

    # Denormalize the fundamental matrix
    F = T2.T @ F @ T1
    print("F error: ", np.sum(F - FOpenCV))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return F


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    # Compute the right epipole (e2): null space of F
    U, S, Vt = np.linalg.svd(F)
    e2 = Vt[-1] + 1e-10 # The last row of V^T, corresponding to the smallest singular value

    # Compute the left epipole (e1): null space of F^T
    U, S, Vt = np.linalg.svd(F.T)
    e1 = U[:, -1] + 1e-10  # The last column of U, corresponding to the smallest singular value

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1, e2




if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task1")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task1", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task1", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task1", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        shape = img1.shape

        # compute F
        F = find_fundamental_matrix(shape, pts1, pts2)
        # compute the epipoles
        e1, e2 = compute_epipoles(F)
        print(e1, e2)
        #to get the real coordinates, divide by the last entry
        print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        outname = os.path.join(output, name + "_us.png")
        # If filename isn't provided or is None, this plt.shows().
        # If it's provided, it saves it
        draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
                      epi1=e1, epi2=e2, filename=outname)



