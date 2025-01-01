"""
bezier
======

Module to model and generate curves using the characteristic matrix implementation of bezier polynomials.

Implements:

1. Generation of cubic bezier curves.
2. Control points of a bezier curve.
3. Curve fit and approximate smooth shapes in the form of cubic splines with single or multiple segments.
4. Control tensor of the entire cubic spline which contains control points for the entire curve.

Contains 2 sub-modules:

1. `cubic` containing the functionality of generating cubic beziers
2. `spline` containing the functionality for generating a complex curve using multi-segment cubic bezier splines

"""