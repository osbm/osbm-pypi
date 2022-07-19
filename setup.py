from setuptools import setup, find_packages


setup(
    name="osbm",
    version=open("osbm/VERSION").read().strip(),
    description="osbm personal library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="osbm",
    author_email="osmanfbayram@gmail.com",
    packages=find_packages(),
    license="MIT",
    url="https://github.com/osbm/osbm-pypi",
    install_requires=open("requirements.txt").read().splitlines(),
    extras_require={
        "chem": open("requirements-chem.txt").read().splitlines()
    },
    python_requires=">=3.7",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)
