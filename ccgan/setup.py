#!/usr/bin/env python

from setuptools import setup, find_packages
import os

long_description = '''
Unsupervised EM-to-ExM Synthesis Using a Structure-Constrained CycleGAN
'''

version = "1.0.0"

requirements = [
]

if __name__ == '__main__':
    setup(
        name='EM2ExM',
        version=version,
        description='EM2ExM image synthesis',
        long_description=long_description,
        author="lauenburg",
        packages=find_packages(
            exclude=[
                'tests',
            ],
            include=[
                'data',
                'models',
                'options',
                'util'
            ],
        ),
        license='MIT',
        install_requires=requirements,
        classifiers=[
          'Development Status :: 1 - Planning',
          'Framework :: PyTorch',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
    )




