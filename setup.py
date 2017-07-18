#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='blinkende_lichter',
    version='0.0.0',
    description='Cell segmentation in 2 photon recordings',
    author='Fabian Sinz, Edgar Walker, Erick Cobos',
    author_email='sinz@bcm.edu',
    url='https://github.com/cajal/blinkende_lichter',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm','gitpython','python-twitter','scikit-image', 'datajoint', 'torch'],
)
