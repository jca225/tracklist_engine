"""Boundary adapters: one per third-party analysis library.

Each module catches only the documented exceptions for its library and
returns a typed domain error from `..errors`. Domain code never imports
the libraries directly.
"""
