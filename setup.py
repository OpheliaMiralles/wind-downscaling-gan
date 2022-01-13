from distutils.core import setup

from setuptools import find_packages

setup(
    name='downscaling',
    version='1.0',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    include_package_data=True,
    package_data={'': ['weights-55.ckpt/*']},
    entry_points={
        'console_scripts': [
            'downscale=downscaling.cli:main',
        ]
    }
)
