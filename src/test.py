"""
Compute geometric features of a 3D point cloud from a .LAS file
Alberto Morcillo Sanz - TIDOP
"""

import computation

def showPercentage(percentage: int) -> None:
    print(str(percentage) + '%')

if __name__ == "__main__":

    # Parameters
    r: float = 5
    path: str = 'C:/Users/EquipoTidop/Desktop/bunny.las'

    # Compute geometric features
    computation.calculateFeatures(path, radius=r, percentageCallback=showPercentage)
