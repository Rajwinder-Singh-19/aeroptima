"""
parser
======

Module to parse .dat files to arrays containing coordinates of the geometry.

Implements:

1. Parsing of selig and lednicer format aerofoil .dat files into coordinates in the form of numpy array
2. Splitting the aerofoil coordinates into upper and lower surfaces for independent control over each surface.

Contains 1 sub-module:
1. `aerofoil` containing parsers for selig and lednicer format .dat files,
along with functionality for splitting upper and lower surface coordinates
"""
