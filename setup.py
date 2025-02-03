from setuptools import setup, find_packages

setup(
    name='vector_perception',
    version='0.1.0',
    author='Alex Lin',
    author_email='alex.lin416@outlook.com',
    description='A utility package for various functions.',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)