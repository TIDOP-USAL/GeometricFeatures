"""
Compute geometric features of a 3D point cloud from a .LAS file
Alberto Morcillo Sanz - TIDOP
"""

import numpy as np
from numpy import linalg as LA
from scipy.spatial import KDTree
import laspy

from features import GeometricFeatures


def getNeighborhoodData(neighborhood: np.array) -> np.array:
    """
    :param neighborhood: an array of points of the neighborhood of the current point
    :return: Matrix with rows X, Y and Z where each row corresponds to a vector of the components x, y, z of each point respectively
    """
    # Separate x, y and z of each neighborhood in 3 different vectors
    X: list[float] = neighborhood[0]
    Y: list[float] = neighborhood[1]
    Z: list[float] = neighborhood[2]

    # Return matrix
    return np.array([X, Y, Z])


def eigen(point: list[float], kdtree: KDTree, radius: float, BIAS=False):
    """
    :param point: Point at which the neighborhood will be calculated
    :param kdtree: KDTree of the pointcloud
    :param radius: Radius of the neighborhood of point
    :param BIAS: BIAS = True means computing the population covariance matrix and BIAS = False, the sample covariance matrix
    :return: Eigenvalues and eigenvectors of the covariance matrix of the neighborhood of a radius 'radius' of the point 'point'
    """
    # Find neighborhood
    idx = kdtree.query_ball_point(point, r=radius)
    neighborhood: np.array = pointcloud[idx]

    # Compute covariance matrix of the neighborhood
    data = getNeighborhoodData(neighborhood)
    covarianceMatrix = np.cov(data, bias=BIAS)

    # Return eigenvalues and eigenvectors
    return LA.eig(covarianceMatrix)


if __name__ == "__main__":

    # Parameters
    r: float = 0.5
    bias: bool = False

    # Load point cloud
    las = laspy.read('c:/Users/EquipoTidop/Desktop/box.las')
    pointcloud = np.array(las.xyz)

    # Build KDTree
    T = KDTree(pointcloud)

    # Calculate the geometric features of each point in a neighborhood of radius r
    for p in pointcloud:

        # Eigenvalues and eigenvectors of the covariance matrix of the neighborhood of p
        eigenvalues, eigenvectors = eigen(p, T, r, bias)

        print('Geometric features of the point', p)

        # Omnivariance
        eigenentropy: float = GeometricFeatures.eigenentropy(eigenvalues)
        print('Eigenentropy of the point:', eigenentropy)

        # Planarity
        planarity: float = GeometricFeatures.planarity(eigenvalues)
        print('Planarity of the point:', planarity)

        print(' ')
