import numpy as np
from matplotlib import pyplot as plt
from homography import fit_homography, homography_transform

def p3(filename: str):
    # code for Task 3
    # 1. load points X from task3/
    X = np.load(filename)
    N, D = X.shape
    
    # 2. fit a transformation y=Sx+t
    X_train = X[:, :2]
    y_train = X[:, 2:]
    
    A = np.zeros((2*N, D+2))
    A[:N, :2] = np.copy(X_train)
    A[N:, 2:4] = np.copy(X_train)
    A[:N, 4] = 1.0
    A[N: , 5] = 1.0
    
    b = np.zeros((2*N, 1))
    b[:N, 0] = y_train[:, 0]
    b[N:, 0] = y_train[:, 1]
    
    Result = np.linalg.lstsq(A, b)
    S = Result[0][:4].reshape(2, 2)
    t = Result[0][4:]
    # print(S, t)
    
    # 3. transform the points
    X_transformed = (S @ X_train.T + t).T
    
    # 4. plot the original points and transformed points
    plt.scatter(X[:, 0], X[:, 1], label='Origianl points', c='blue', s=1)
    plt.scatter(X[:, 2], X[:, 3], label='Transformed GT', c='red', s=1)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], label='Transformed points', c='green', s=1.5)
    plt.legend()
    plt.show()
    
    return S, t

def p4(filename: str):
    # code for Task 4
    XY = np.load(filename)
    N = XY.shape[0]
    X_train = XY[:, :2]
    y_train = XY[:, 2:]
    H = fit_homography(XY)
    
    X_transformed = (H @ (np.hstack((X_train, np.ones((N, 1))))).T).T
    X_transformed /= X_transformed[:, [2]] #Check this
    
    plt.scatter(XY[:, 0], XY[:, 1], label='Origianl points', c='blue', s=1)
    plt.scatter(XY[:, 2], XY[:, 3], label='Transformed GT', c='red', s=1)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], label='Transformed points', c='green', s=1.5)
    plt.legend()
    plt.show()
    
    return H

if __name__ == "__main__":
    # Task 3
    # S1, t1 = p3("task3/points_case_1.npy")
    # print(S1, t1)
    # S2, t2 = p3("task3/points_case_2.npy")
    # print(S2, t2)

    # Task 4
    H1 = p4("task4/points_case_1.npy")
    print(H1)
    H4 = p4("task4/points_case_4.npy")
    print(H4)
    p4("task4/points_case_5.npy")
    p4("task4/points_case_9.npy")