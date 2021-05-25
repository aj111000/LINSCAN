# -*- coding: utf-8 -*-
"""
This is a simple implementation of DBSCAN intended to explain the algorithm.
@author: Chris McCormick
"""

import numpy as nnp
from scipy.spatial import KDTree


def ecc(ptindex_array, eps):
    pt_array = D[ptindex_array]
    center_pt = pt_array.mean(axis=0)
    pt_trans = pt_array - center_pt
    tempA = np.square(pt_trans)
    A = sum(tempA[:, 0]) - sum(tempA[:, 1])
    C = 2 * sum(pt_trans[:, 0] * pt_trans[:, 1])
    B = (A ** 2 + C ** 2) ** 0.5

    if (C == 0):
        if (-A + B == 0):
            angle = 0
        else:
            angle = math.pi / 2
    else:
        angle = math.atan((-A + B) / C)

    a_ori = math.sqrt(
        sum(np.square(pt_trans[:, 1] * math.sin(angle) + pt_trans[:, 0] * math.cos(angle))) / len(pt_array))
    b_ori = math.sqrt(
        sum(np.square(pt_trans[:, 1] * math.cos(angle) - pt_trans[:, 0] * math.sin(angle))) / len(pt_array))

    if a_ori * b_ori == 0:
        if a_ori > b_ori:
            a_final = float('inf')
            b_final = 0
        else:
            b_final = float('inf')
            a_final = 0
    else:
        trans_indicator = math.sqrt(eps ** 2 / (a_ori * b_ori))
        if a_ori > b_ori:
            a_final = a_ori * trans_indicator
            b_final = b_ori * trans_indicator
        else:
            b_final = a_ori * trans_indicator
            a_final = b_ori * trans_indicator

    result = SDE(center_pt, a_final, b_final, angle)
    return result


def line_dbscan(D, eps, min_pts):
    """
    Cluster the dataset `D` using the DBSCAN algorithm.

    MyDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.

    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    """

    kd = KDTree(D)
    eccentricities = [ecc(kd.query(D[i], k=min_pts)[1], eps) for i in range(len(D))]

    # This list will hold the final cluster assignment for each point in D.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.    
    labels = [0] * len(D)

    # C is the ID of the current cluster.    
    C = 0

    # This outer loop is just responsible for picking new seed points--a point
    # from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'expandCluster' routine.

    # For each point P in the Dataset D...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)
    for P in range(0, len(D)):

        # Only points that have not already been claimed can be picked as new 
        # seed points.    
        # If the point's label is not 0, continue to the next point.
        if not (labels[P] == 0):
            continue

        # Find all of P's neighboring points.
        NeighborPts = region_query(D, P, eps)

        # If the number is below MinPts, this point is noise. 
        # This is the only condition under which a point is labeled 
        # NOISE--when it's not a valid seed point. A NOISE point may later 
        # be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to 
        # something else).
        if len(NeighborPts) < min_pts:
            labels[P] = -1
        # Otherwise, if there are at least MinPts nearby, use this point as the 
        # seed for a new cluster.    
        else:
            C += 1
            grow_cluster(D, labels, P, NeighborPts, C, eps, min_pts)

    # All data has been clustered!
    return labels


def grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts):
    """
    Grow a new cluster with label `C` from the seed point `P`.

    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.

    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """

    # Assign the cluster label to the seed point.
    labels[P] = C

    # Look at each neighbor of P (neighbors are referred to as Pn). 
    # NeighborPts will be used as a FIFO queue of points to search--that is, it
    # will grow as we discover new branch points for the cluster. The FIFO
    # behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPts, the points are represented by their index in the original
    # dataset.
    i = 0
    while i < len(NeighborPts):

        # Get the next point from the queue.        
        Pn = NeighborPts[i]

        # If Pn was labelled NOISE during the seed search, then we
        # know it's not a branch point (it doesn't have enough neighbors), so
        # make it a leaf point of cluster C and move on.
        if labels[Pn] == -1:
            labels[Pn] = C

        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C

            # Find all the neighbors of Pn
            PnNeighborPts = region_query(D, Pn, eps)

            # If Pn has at least MinPts neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched. 
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
            # If Pn *doesn't* have enough neighbors, then it's a leaf point.
            # Don't queue up it's neighbors as expansion points.
            # else:
            # Do nothing
            # NeighborPts = NeighborPts

        # Advance to the next point in the FIFO queue.
        i += 1

        # We've finished growing cluster C!


def region_query(D, P, eps):
    """
    Find all points in dataset `D` within distance `eps` of point `P`.

    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    """
    neighbors = []

    # For each point in the dataset...
    for Pn in range(0, len(D)):

        # If the distance is below the threshold, add it to the neighbors list.
        if numpy.linalg.norm(D[P] - D[Pn]) < eps:
            neighbors.append(Pn)

    return neighbors