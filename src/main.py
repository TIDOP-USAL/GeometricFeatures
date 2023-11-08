"""
Compute geometric features of a 3D point cloud from a .LAS file
Alberto Morcillo Sanz - TIDOP
"""

import numpy as np
from numpy import linalg as LA

from scipy.spatial import KDTree

import laspy

from features import GeometricFeatures

SCALAR_FIELDS = ['Sum_of_eigenvalues', 'Omnivariance', 'Eigenentropy',
                 'Anisotropy', 'Linearity', 'Planarity', 'Sphericity',
                 'PCA1', 'PCA2', 'Surface_variation', 'Verticality'
                 ]

def addScalarFields(las: laspy.LasData, radius: float) -> None:
    """
    :param las: LAS file
    :param radius: Neighborhood radius
    :return:
    """
    for scalarField in SCALAR_FIELDS:

        las.add_extra_dim(laspy.ExtraBytesParams(
            name=scalarField,
            type=np.float64,
        ))


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


def eigen(point: list[float], kdtree: KDTree, pointcloud: np.array, radius: float, BIAS=False):
    """
    :param point: Point at which the neighborhood will be calculated
    :param kdtree: KDTree of the pointcloud previously calculated
    :param pointcloud: The point cloud
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


def calculateFeatures(cloudPath: str, radius: float, bias: bool = False) -> None:
    """
    :param cloudPath:
    :param radius:
    :param bias:
    :return:
    """
    # Load point cloud
    las = laspy.read(cloudPath)
    addScalarFields(las, radius)
    print(list(las.point_format.dimension_names))

    # Build kdtree
    pointcloud = np.array(las.xyz)
    T = KDTree(pointcloud)

    # Calculate the geometric features of each point in a neighborhood of radius r
    print('Calculating geometric features...')
    pointIndex: int = 0
    for p in pointcloud:

        # Eigenvalues and eigenvectors of the covariance matrix of the neighborhood of p
        eigenvalues, eigenvectors = eigen(p, T, pointcloud, r, bias)

        # Sum of eigenvalues
        sumofeigenvalues: float = GeometricFeatures.sumOfEigenValues(eigenvalues)
        las[SCALAR_FIELDS[0]][pointIndex] = sumofeigenvalues

        # Omnivariance
        omnivariance: float = GeometricFeatures.omnivariance(eigenvalues)
        las[SCALAR_FIELDS[1]][pointIndex] = omnivariance

        # Eigenentropy
        eigenentropy: float = GeometricFeatures.eigenentropy(eigenvalues)
        las[SCALAR_FIELDS[2]][pointIndex] = eigenentropy

        # Anisotropy
        anisotropy: float = GeometricFeatures.anisotropy(eigenvalues)
        las[SCALAR_FIELDS[3]][pointIndex] = anisotropy

        # Linearity
        linearity: float = GeometricFeatures.linearity(eigenvalues)
        las[SCALAR_FIELDS[4]][pointIndex] = linearity

        # Planarity
        planarity: float = GeometricFeatures.planarity(eigenvalues)
        las[SCALAR_FIELDS[5]][pointIndex] = planarity

        # Sphericity
        sphericity: float = GeometricFeatures.sphericity(eigenvalues)
        las[SCALAR_FIELDS[6]][pointIndex] = sphericity

        # PCA1
        PCA1: float = GeometricFeatures.PCA1(eigenvalues)
        las[SCALAR_FIELDS[7]][pointIndex] = PCA1

        # PCA2
        PCA2: float = GeometricFeatures.PCA2(eigenvalues)
        las[SCALAR_FIELDS[8]][pointIndex] = PCA2

        # Surface variation
        surfaceVariation: float = GeometricFeatures.surfaceVariation(eigenvalues)
        las[SCALAR_FIELDS[9]][pointIndex] = surfaceVariation

        # Verticality (check if the point cloud has normals)
        pointIndex += 1

    # Save point cloud
    print('Saving point cloud...')
    outputPath: str = cloudPath.split('.')[0] + '-new.las'
    las.write(outputPath)

if __name__ == "__main__":

    # Parameters
    r: float = 0.5
    b: bool = False
    path: str = 'c:/Users/EquipoTidop/Desktop/box.las'

    calculateFeatures(path, radius=r, bias=b)
