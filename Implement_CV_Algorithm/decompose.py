import numpy as np
from triangulate import triangulate_all_points
import cv2

"""
## estimate the R, t of the camera given the essential matrix E
## see: Richard Hartley and Andrew Zisserman (2003). Multiple View Geometry in computer vision
## http://isit.u-clermont1.fr/~ab/Classes/DIKU-3DCV2/Handouts/Lecture16.pdf
"""


def decompose(E):

    '''
    ## TODO 4.1
    ## Decompose the essential matrix E into R1, R2, t1, t2.
    ## we will get four [R t] combination and only one is possible
    '''

    U, S, V = np.linalg.svd(E, full_matrices=True)
    W = np.array([0, -1, 0, 1, 0, 0, 0, 0, 1]).reshape((3, 3))

    # scale s
    s = U[:, 2] / cv2.norm(U[:, 2])

    R1 = np.dot(U, W).dot(V)
    R2 = np.dot(U, W.T).dot(V)
    t1 = s
    t2 = -s


    return R1, R2, t1, t2


def relativeTransformation(E, points1, points2, K):

    R1, R2, t1, t2 = decompose(E)
    ## A negative determinant means that R contains a reflection.This is not rigid transformation!
    if np.linalg.det(R1) < 0:
        E = -E
        R1, R2, t1, t2 = decompose(E)

    bestCount = 0

    for dR in range(2):
        if dR == 0:
            cR = R1
        else:
            cR = R2
        for dt in range(2):
            if dt == 0:
                ct = t1
            else:
                ct = t2

            View1 = np.eye(3, 4)
            View2 = np.zeros((3, 4))
            for i in range(3):
                for j in range(3):
                    View2[i, j] = cR[i, j]
            for i in range(3):
                View2[i, 3] = ct[i]

            count = len(triangulate_all_points(View1, View2, K, points1, points2))
            if (count > bestCount):
                V = View2

        cR = np.transpose(cR)

    print ("V " + str(V))
    return V


def testDecompose():
    E = np.array([[3.193843825323984,  1.701122615578195,  -6.27143074201245],
                  [-41.95294882825382, 127.76771801644763, 141.5226870527082],
                  [-8.074440557013252, -134.9928434422784, 1]])
    R1, R2, t1, t2 = decompose(E)

    print("Testing decompose...")

    t_ref = np.array([-0.9973654079783344, -0.04579551552933408, -0.05625845470338976])

    if np.linalg.norm(t1 - t_ref) < 1e-5:
        if np.linalg.norm(t2 + t_ref) < 1e-5:
            print("Test (Translation): SUCCESS")
        else:
            print("Test (Translation): FAIL!")
    elif np.linalg.norm(t1 + t_ref) < 1e-5:
        if np.linalg.norm(t2 - t_ref) < 1e-5:
            print("Test (Translation): SUCCESS")
        else:
            print("Test (Translation): FAIL!")

    R1_ref = np.array([[0.9474295893819155,  -0.1193419720472381, 0.2968748336782551],
                       [0.2288481582901039, 0.9012012887804273, -0.3680553729369057],
                       [-0.2236195286884464, 0.4166458097813701, 0.8811359574894123]])
    R2_ref = np.array([[0.9332689072231527,  0.01099472166665292, 0.3590101153254257],
                       [-0.1424930907107563, -0.9061762138657301, 0.3981713055001164],
                       [0.329704209724715,   -0.4227573601008549, -0.8441394129942975]])

    if np.linalg.norm(R1-R1_ref) < 1e-5:
        if np.linalg.norm(R2-R2_ref) < 1e-5:
            print("Test (Rotation): SUCCESS!")
        else:
            print("Test (Rotation): FAIL!")
    elif np.linalg.norm(R1 - R2_ref) < 1e-5:
        if np.linalg.norm(R2 - R1_ref) < 1e-5:
            print("Test (Rotation): SUCCESS!")
        else:
            print("Test (Rotation): FAIL!")
    print ('='*25)