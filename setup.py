from setuptools import setup, find_packages

VERSION = "0.0.2"

setup(
    name='osbm',
    version=VERSION,
    description='osbm personal library',
    long_description=open('README.md').read(),
    author='osbm',
    author_email="osmanfbayram@gmail.com",
    packages=find_packages(),
    license="MIT",
    url="https://github.com/osbm/osbm-pypi",
    # read the requirements.txt file
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires=">=3.7",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        ],

)