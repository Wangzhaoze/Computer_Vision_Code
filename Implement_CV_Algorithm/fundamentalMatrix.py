import numpy as np


def epipolarConstraint(p1, p2, F, t):

    p1h = np.array([p1[0], p1[1], 1])
    p2h = np.array([p2[0], p2[1], 1])

    
    ## 1.Compute the normalized epipolar line
    ## 2.Compute the distance to the epipolar line
    ## 3.Check if the distance is smaller than threshold
    
    l = np.dot(F, p1h)
    l = l / np.sqrt(p1[0]**2 + p1[1]**2)
    d = np.dot(p2h.T, l)
    if abs(d) < t:
        return True
    else:
        return False


def computeF(points1, points2):

    assert (len(points1) == 8), "Length of points1 should be 8!"
    assert (len(points2) == 8), "Length of points2 should be 8!"

    A = np.zeros((8, 9)).astype('int')
    for i in range(8):

        # fill up A matrix
        (px, py) = points1[i]
        (qx, qy) = points2[i]
        A[i, :] = [px*qx, px*qy, px, py*qx, py*qy, py, qx, qy, 1]


    ## Solve linear estimation Af = 0 with SVD.1
    ## pay attention to reshape method from vector h to matrix H

    U, s, V = np.linalg.svd(A, full_matrices=True)
    F = V.T[:, -1].reshape((3, 3)).T




    ## Enforce Rank(F) = 2 by SVD.2
    ## normalize F and return

    UF, sF, VF = np.linalg.svd(F)
    sF[2] = 0
    F = np.dot(np.dot(UF, np.diag(sF)), VF)
    F = F * (1.0 / F[2, 2])

    return F



def numInliers(points1, points2, F, threshold):
    inliers = []
    for i in range(len(points1)):
        if (epipolarConstraint(points1[i], points2[i], F, threshold)):
            inliers.append(i)

    return inliers


def computeFRANSAC(points1, points2):

    ## The best fundamental matrix and the number of inlier for this F.
    bestInlierCount = 0
    threshold = 4
    iterations = 1000

    for k in range(iterations):
        subset1 = []
        subset2 = []
        for i in range(8):
            x = np.random.randint(0, len(points1)-1)
            subset1.append(points1[x])
            subset2.append(points2[x])
        F = computeF(subset1, subset2)
        num = numInliers(points1, points2, F, threshold)
        if (len(num) > bestInlierCount):
            bestF = F
            bestInlierCount = len(num)
            bestInliers = num

    return (bestF, bestInliers)

def testFundamentalMat():
    points1 = [(1, 1), (3, 7), (2, -5), (10, 11), (11, 2), (-3, 14), (236, -514), (-5, 1)]
    points2 = [(25, 156), (51, -83), (-144, 5), (345, 15),
                                    (215, -156), (151, 83), (1544, 15), (451, -55)]

    F = computeF(points1, points2)

    print ("Testing Fundamental Matrix...")
    print ("Your result:" + str(F))

    Href = np.array([[0.001260822171230067,  0.0001614643951166923, -0.001447955678643285],
                 [-0.002080014358205309, -0.002981504896782918, 0.004626528742122177],
                 [-0.8665185546662642,   -0.1168790312603214,   1]])

    print ("Reference: " + str(Href))

    error = Href - F
    e = np.linalg.norm(error)
    print ("Error: " + str(e))

    if (e < 1e-10):
        print ("Test: SUCCESS!")
    else:
        print ("Test: FAIL!")
    print ("============================")
