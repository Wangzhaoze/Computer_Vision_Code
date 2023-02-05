import cv2
import numpy as np


def triangulate(M1, M2, p1, p2):
    """
    ## Triangulation Algorithm
    ## find homogeneous coordinate of a point x in 3d space
    ## given its position p, q in two view images and the camera parameters M of both views
    ## pay attention to len of coordinate vector and remember normalization after projection
    """

    '''
    ## TODO 3.1
    ## - Construct the matrix A'''
    (px, py) = p1
    (qx, qy) = p2
    A = [px * M1[2] - M1[0], py * M1[2] - M1[1], qx * M2[2] - M2[0], qy * M2[2] - M2[1]]

    # solve linear estimation of Ax = 0
    U, s, V = np.linalg.svd(A, full_matrices=True)

    '''
    ## TODO 3.2
    ## Extract the solution and project it back to 3D (from homogenous space)
    ## x_homo [X, Y, Z, W] = V.T[:, -1]
    ## x_3d = [X/W, Y/W, Z/W]
    '''

    x = V[-1, 0:3] / V[-1, 3]

    return x


def triangulate_all_points(View1, View2, K, points1, points2):
    wps = []
    P1 = np.dot(K, View1)
    P2 = np.dot(K, View2)

    for i in range(len(points1)):
        wp = triangulate(P1, P2, points1[i], points2[i])

        ## Check if this points is in front of both cameras
        ptest = [wp[0], wp[1], wp[2], 1]
        p1 = np.matmul(P1, ptest)
        p2 = np.matmul(P2, ptest)

        if (p1[2] > 0 and p2[2] > 0):
            wps.append(wp)

    return wps


def testTriangulate():
    P1 = np.array([[1, 2, 3, 6], [4, 5, 6, 37], [7, 8, 9, 15]]).astype('float')
    P2 = np.array([[51, 12, 53, 73], [74, 15, -6, -166], [714, -8, 95, 16]]).astype('float')

    F = triangulate(P1, P2, (14.0, 267.0), (626.0, 67.0))
    print("Testing Triangulation...")
    print("Your result: " + str(F))

    wpref = [0.782409, 3.89115, -5.72358]
    print("Reference: " + str(wpref))

    error = wpref - F
    e = cv2.norm(error)
    print("Error: " + str(e))

    if (e < 1e-5):
        print("Test: SUCCESS!")
    else:
        print("Test: FAIL")
    print("================================")
