import pdb
import cv2
from utils import createMatchImage
import numpy as np


def matchknn2(descriptors1, descriptors2):
    """
    ## knn match ratio algorithm
    ## find first and second closest points in descriptors of feature points from to images
    ## return list of objects which contains all matched point pairs
    ## these objects could be defined by cv2.DMatch
    """

    ## Initialize an empty list of matches. (HINT: N x 2)
    knnmatches = []

    ## TODO 2.1
    ## Find the two nearest neighbors for every descriptor in image 1.
    n1, n2 = descriptors1.shape[0], descriptors2.shape[0]

    for i in range(n1):
        distances = list()
        for j in range(n2):
            # compute all distances of one to many points
            distances.append(cv2.norm(descriptors1[i], descriptors2[j], cv2.NORM_HAMMING))

        # sort results of distances
        sorted_j = np.argsort(distances)
        distances.sort()

        # define DMatch objects
        dm1 = cv2.DMatch(i, sorted_j[0], distances[0])
        dm2 = cv2.DMatch(i, sorted_j[1], distances[1])

        knnmatches.append([dm1, dm2])

        knnmatches = np.array(knnmatches).reshape(len(knnmatches), 2).tolist()


    return knnmatches


def ratioTest(knnmatches, ratio_threshold):
    """
    ## TODO 2.2
    ## Compute the ratio between the nearest and second nearest neighbor.
    ## Add the nearest neighbor to the output matches if the ratio is smaller than ratio_threshold.
    """

    matches = []

    for i in range(len(knnmatches)):
        dis1 = knnmatches[i][0].distance
        dis2 = knnmatches[i][1].distance
        if dis1 / dis2 < ratio_threshold:
            matches.append(knnmatches[i][0])

    return matches



def computeMatches(img1, img2):
    knnmatches = matchknn2(img1['descriptors'], img2['descriptors'])
    matches = ratioTest(knnmatches, 0.7)
    print ("(" + str(img1['id']) + "," + str(img2['id']) + ") found " + str(len(matches)) + " matches.")
    return matches
