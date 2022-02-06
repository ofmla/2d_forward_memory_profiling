"""
This module produces shot gathers in segy format. Acoustic, viscoacoustic and anisotropic
wave equations for TTI media can be used. Linearized Born modelling is also supported.
"""
from dask_cluster import DaskCluster


class ControlGetshot:
    """
    Class to generate the shot files based on the input coordinates or
    geometry parameters files
    """

    def generate_shot_files(self):
        "Fetches the input from files and starts to generate the shots"
        dask_cluster = DaskCluster()
        dask_cluster.generate_shots_in_cluster()
