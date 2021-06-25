from distutils.core import setup

from setuptools import find_packages

setup(
    name='downscaling',
    version='1.0',
    package_dir={'': 'src'},
    packages=find_packages('src'),
)
