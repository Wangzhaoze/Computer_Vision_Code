import cv2
import numpy as np
from homography import computeHomography


def numInliers(points1, points2, H, threshold):

    """
    ## TODO 4.1
    ## Compute the number of inliers for the given homography and corresponding points
    ## Project the points from image 1 to image 2
    ## point is an inlier if the distance between the projected point and the point in image 2 is smaller than threshold
    ## Hint: Construct point 1 in Homogeneous coordinate before applying H.
    """

    InlierCount = 0
    n = len(points1)

    for i in range(n):

        q = list(points2[i])
        p = list(points1[i])

        # convert in homogeneous coordinate
        q.append(1)
        p.append(1)

        # project in 2nd img with given H
        p_ = np.dot(H, np.array(p))

        # normalize projected p
        p_ /= p_[2]

        # compute distance and compare with threshold
        dis = cv2.norm(np.array(q) - p_)
        if dis < threshold:
            InlierCount += 1

    return InlierCount


def computeHomographyRansac(img1, img2, matches, iterations, threshold):
    '''
    reduce outliers in matching by RANSAC Algorithm
    :param img1 & img2: input two images
    :param matches: retured list of matching points from knnmatch()
    :param iterations: iteration times
    :param threshold: threshold
    :return: the best result of Homography Matrix with the most inliers
    '''

    bestInlierCount = 0
    bestH = []
    # prepare data of matching points
    points1 = []
    points2 = []
    for i in range(len(matches)):
        points1.append(img1['keypoints'][matches[i].queryIdx].pt)
        points2.append(img2['keypoints'][matches[i].trainIdx].pt)

    for i in range(iterations):

        ## TODO 4.2
        ## Construct the subsets by randomly choosing 4 matches.
        subset1 = []
        subset2 = []
        for j in range(4):
            x = np.random.randint(0, len(points1) - 1)
            subset1.append(points1[x])
            subset2.append(points2[x])

        ## Compute the homography for this subset
        temp_H = computeHomography(subset1, subset2)

        ## Compute the number of inliers by applying H into all matching points but not subset
        temp_num = numInliers(points1, points2, temp_H, threshold=threshold)

        ## Keep track of the best homography (use the variables bestH and bestInlierCount)
        if bestInlierCount < temp_num:
            bestInlierCount = temp_num
            bestH = temp_H

    print ("(" + str(img1['id']) + "," + str(img2['id']) + ") found " + str(bestInlierCount) + " RANSAC inliers.")

    return bestH
