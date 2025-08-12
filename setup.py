"""
Setup configuration for CellSAM
"""

import os

from setuptools import find_packages, setup


# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="cellsam",
    version="1.0.0",
    author="CellSAM Development Team",
    author_email="",
    description="Standalone Cell Segmentation with Segment Anything Model",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "cellsam=cellsam.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cellsam": ["*.md", "*.txt"],
    },
    keywords="cell segmentation, microscopy, deep learning, computer vision, SAM",
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
)
