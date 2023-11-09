"""
Compute geometric features of a 3D point cloud from a .LAS file
Alberto Morcillo Sanz - TIDOP
"""

import numpy as np
from numpy import log as ln

import math
import sys

class GeometricFeatures:
    """
    Contains all the methods for calculating the geometric features
    of a neighborhood of a 3D point cloud
    """

    @staticmethod
    def sumOfEigenValues(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\sum_{i} \lambda_{i}$
        """
        return eigenvalues[0] + eigenvalues[1] + eigenvalues[2]

    @staticmethod
    def omnivariance(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\left( \prod_{i} \lambda_{i} \right)^\frac{1}{3}$
        """
        return math.pow(eigenvalues[0] * eigenvalues[1] * eigenvalues[2], 1.0 / 3.0)

    @staticmethod
    def eigenentropy(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $-\sum_{i}\lambda_{i}\ln\left( \lambda_{i} \right )$
        """
        sum: float = 0
        for eigenvalue in eigenvalues:
            if eigenvalue <= 0:  # ln(x) = -inf, x <= 0. As we have -sum, -(-inf) = inf
                return float('inf')
            sum += eigenvalue * ln(eigenvalue)
        return -sum

    @staticmethod
    def anisotropy(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\frac{\lambda_{1} - \lambda_{3}}{\lambda_{1}}$
        """
        return (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0] if abs(eigenvalues[0]) > sys.float_info.epsilon else float('NaN')

    @staticmethod
    def linearity(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\frac{\lambda_{1} - \lambda_{2}}{\lambda_{1}}$
        """
        return (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if abs(eigenvalues[0]) > sys.float_info.epsilon else float('NaN')

    @staticmethod
    def planarity(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\frac{\lambda_{2} - \lambda_{3}}{\lambda_{1}}$
        """
        return (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if abs(eigenvalues[0]) > sys.float_info.epsilon else float('NaN')

    @staticmethod
    def sphericity(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\frac{\lambda_{3}}{\lambda_{1}}$
        """
        return eigenvalues[2] / eigenvalues[0] if abs(eigenvalues[0]) > sys.float_info.epsilon else float('NaN')

    @staticmethod
    def PCA1(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\lambda_{1}\left ( \sum _{i} \lambda_{i} \right )^{-1}$
        """
        eigenvaluesSum = GeometricFeatures.sumOfEigenValues(eigenvalues)
        return eigenvalues[0] / eigenvaluesSum if abs(eigenvaluesSum) > sys.float_info.epsilon else float('NaN')

    @staticmethod
    def PCA2(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\lambda_{2}\left ( \sum _{i} \lambda_{i} \right )^{-1}$
        """
        eigenvaluesSum = GeometricFeatures.sumOfEigenValues(eigenvalues)
        return eigenvalues[1] / eigenvaluesSum if abs(eigenvaluesSum) > sys.float_info.epsilon else float('NaN')

    @staticmethod
    def surfaceVariation(eigenvalues: np.array) -> float:
        """
        :param eigenvalues: Eigenvalues of the covariance matrix of a neighborhood
        :return: $\lambda_{3}\left ( \sum _{i} \lambda_{i} \right )^{-1}$
        """
        eigenvaluesSum = GeometricFeatures.sumOfEigenValues(eigenvalues)
        return eigenvalues[2] / eigenvaluesSum if abs(eigenvaluesSum) > sys.float_info.epsilon else float('NaN')

    @staticmethod
    def verticality(eigenvectors: np.array) -> float:
        """
        :param eigenvectors: Eigenvectors associated to the eigenvalues of the covariance matrix of a neighborhood
        :return: $1 - \left | n_{z} \right |$
        """
        z = [0, 0, 1]
        e3 = eigenvectors[2]
        return 1.0 - math.fabs(np.dot(z, e3))

