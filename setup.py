#!/usr/bin/env python

from setuptools import setup

setup(name='torch_nf',
      version='0.1',
      description='Normalizing flows in PyTorch.',
      author='Sean Bittner',
      author_email='seanrbittner@gmail.com',
      install_requires=['torch',
                        'numpy',
                        'pytest-cov'],
      packages=['torch_nf'],
     )
