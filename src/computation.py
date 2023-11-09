"""
Compute geometric features of a 3D point cloud from a .LAS file
Alberto Morcillo Sanz - TIDOP
"""

import numpy as np
from numpy import linalg as LA

from scipy.spatial import KDTree

import laspy

from features import GeometricFeatures

SCALAR_FIELDS: list[str] = ['Sum_of_eigenvalues', 'Omnivariance', 'Eigenentropy',
                 'Anisotropy', 'Linearity', 'Planarity', 'Sphericity',
                 'PCA1', 'PCA2', 'Surface_variation', 'Verticality'
                 ]

MIN_NEIGHBORHOOD: int = 3

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
    X: list[float] = []
    Y: list[float] = []
    Z: list[float] = []

    for neighbor in neighborhood:
        X.append(neighbor[0])
        Y.append(neighbor[1])
        Z.append(neighbor[2])

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

    if len(neighborhood) < MIN_NEIGHBORHOOD:
        raise RuntimeError()

    # Compute covariance matrix of the neighborhood
    data = getNeighborhoodData(neighborhood)
    covarianceMatrix = np.cov(data, bias=BIAS)

    # Return eigenvalues and eigenvectors
    return LA.eig(covarianceMatrix)

def computeFeatures(las: laspy.LasData, eigenvalues: np.array, eigenvectors: np.array, idx: int) -> None:
    """
    :param las: LAS file
    :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
    :param eigenvectors: Eigenvectors associated to the eigenvalues of the covariance matrix of a neighborhood
    :param idx: Scalarfield index
    :return:
    """
    # Sum of eigenvalues
    sumofeigenvalues: float = GeometricFeatures.sumOfEigenValues(eigenvalues)
    las[SCALAR_FIELDS[0]][idx] = sumofeigenvalues

    # Omnivariance
    omnivariance: float = GeometricFeatures.omnivariance(eigenvalues)
    las[SCALAR_FIELDS[1]][idx] = omnivariance

    # Eigenentropy
    eigenentropy: float = GeometricFeatures.eigenentropy(eigenvalues)
    las[SCALAR_FIELDS[2]][idx] = eigenentropy

    # Anisotropy
    anisotropy: float = GeometricFeatures.anisotropy(eigenvalues)
    las[SCALAR_FIELDS[3]][idx] = anisotropy

    # Linearity
    linearity: float = GeometricFeatures.linearity(eigenvalues)
    las[SCALAR_FIELDS[4]][idx] = linearity

    # Planarity
    planarity: float = GeometricFeatures.planarity(eigenvalues)
    las[SCALAR_FIELDS[5]][idx] = planarity

    # Sphericity
    sphericity: float = GeometricFeatures.sphericity(eigenvalues)
    las[SCALAR_FIELDS[6]][idx] = sphericity

    # PCA1
    PCA1: float = GeometricFeatures.PCA1(eigenvalues)
    las[SCALAR_FIELDS[7]][idx] = PCA1

    # PCA2
    PCA2: float = GeometricFeatures.PCA2(eigenvalues)
    las[SCALAR_FIELDS[8]][idx] = PCA2

    # Surface variation
    surfaceVariation: float = GeometricFeatures.surfaceVariation(eigenvalues)
    las[SCALAR_FIELDS[9]][idx] = surfaceVariation

    # Verticality
    verticality: float = GeometricFeatures.verticality(eigenvectors)
    las[SCALAR_FIELDS[10]][idx] = verticality

def calculateFeatures(cloudPath: str, radius: float, bias: bool = False, percentageCallback=None) -> None:
    """
    :param cloudPath: Path of the point cloud
    :param radius: Neighborhood radius
    :param bias: Set bias to false to compute the sample covariance or set it to true to compute the population covariance
    :param percentageCallback: [OPTIONAL] calls a callback for dealing with the percentage of the computation
    :return: None
    """
    # Load point cloud
    las = laspy.read(cloudPath)
    addScalarFields(las, radius)

    numPoints: int = len(las.points)

    # Build kdtree
    pointcloud = np.array(las.xyz)
    T = KDTree(pointcloud)

    # Calculate the geometric features of each point in a neighborhood of radius r
    pointIndex: int = 0
    for p in pointcloud:

        # Eigenvalues and eigenvectors of the covariance matrix of the neighborhood of p
        try:
            eigenvalues, eigenvectors = eigen(p, T, pointcloud, radius, bias)
        except RuntimeError:
            print("Couldn't compute covariance matrix in neighborhoods smaller than 3 points. Consider a bigger radius")
            continue

        # Compute geometric features
        computeFeatures(las, eigenvalues, eigenvectors, pointIndex)
        pointIndex += 1

        # Compute optional percentage
        if percentageCallback is not None:
            percentage = max(int(100 * float(pointIndex) / numPoints) - 1, 0)
            percentageCallback(percentage)

    # Save point cloud
    outputPath: str = cloudPath.split('.')[0] + '-features.las'
    las.write(outputPath)

    # Compute optional percentage
    if percentageCallback is not None:
        percentageCallback(100)
