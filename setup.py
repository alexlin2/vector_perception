from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vector_perception'

setup(
    name=package_name,
    version='0.1.0',
    author='Alex Lin',
    author_email='alex.lin416@outlook.com',
    description='A utility package for various functions.',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    package_data={
        'vector_perception.segmentation': ['config/*.yaml'],
        'vector_perception.detection2d': ['config/*.yaml'],
    },
    include_package_data=True,
)