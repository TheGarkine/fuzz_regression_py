# coding: utf-8

"""
    Fuzzy Regression
"""


from setuptools import setup, find_packages  # noqa: H301

NAME = "fuzzy_regression"
VERSION = "0.0.2"
# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = ["cvxopt", "numpy", "matplotlib"]

setup(
    name=NAME,
    version=VERSION,
    description="Fuzzy Regression Library",
    author_email="krauthannra64754@th-nuernberg.de",
    url="th-nuernberg.de",
    keywords=["Fuzzy", "Regression"],
    install_requires=REQUIRES,
    packages=find_packages(),
    include_package_data=True,
    long_description="""\
    Library that implements several approaches for Fuzzy Regression
    """
)