import numpy as np
import cv2


def computeHomography(points1, points2):
    """
    Compute a homography matrix from 4 point matches -- 8 points algorithm
    points1: [(1, 1), (3, 7), (2, -5), (10, 11)]
    points2: [(25, 156), (51, -83), (-144, 5), (345, 15)]
    """
    assert (len(points1) == 4)
    assert (len(points2) == 4)

    [(px1, py1), (px2, py2), (px3, py3), (px4, py4)] = points1
    [(qx1, qy1), (qx2, qy2), (qx3, qy3), (qx4, qy4)] = points2

    
    ## Construct A matrix used for linear least square estimation 
    

    A = [
        [-px1, -py1, -1, 0, 0, 0, px1 * qx1, py1 * qx1, qx1],
        [0, 0, 0, -px1, -py1, -1, px1 * qy1, py1 * qy1, qy1],
        [-px2, -py2, -1, 0, 0, 0, px2 * qx2, py2 * qx2, qx2],
        [0, 0, 0, -px2, -py2, -1, px2 * qy2, py2 * qy2, qy2],
        [-px3, -py3, -1, 0, 0, 0, px3 * qx3, py3 * qx3, qx3],
        [0, 0, 0, -px3, -py3, -1, px3 * qy3, py3 * qy3, qy3],
        [-px4, -py4, -1, 0, 0, 0, px4 * qx4, py4 * qx4, qx4],
        [0, 0, 0, -px4, -py4, -1, px4 * qy4, py4 * qy4, qy4]]

    # SVD
    U, s, V = np.linalg.svd(A, full_matrices=True)
    V = np.transpose(V)


    ## - Extract the homogeneous solution of Ah=0 as the last column vector of V.
    ## - Store the result in H and Normalize H with h9

    H = V[:, 8].reshape((3, 3))
    H = H / H[2, 2]

    return H


def testHomography():
    points1 = [(1, 1), (3, 7), (2, -5), (10, 11)]
    points2 = [(25, 156), (51, -83), (-144, 5), (345, 15)]

    H = computeHomography(points1, points2)

    print("Testing Homography...")
    print("Your result:" + str(H))

    Href = np.array([[-151.2372466105457, 36.67990057507507, 130.7447340624461],
                     [-27.31264543681857, 10.22762978292494, 118.0943169422209],
                     [-0.04233528054472634, -0.3101691983762523, 1]])

    print("Reference: " + str(Href))

    error = Href - H
    e = np.linalg.norm(error)
    print("Error: " + str(e))

    if (e < 1e-10):
        print("Test: SUCCESS!")
    else:
        print("Test: FAIL!")
    print("============================")
