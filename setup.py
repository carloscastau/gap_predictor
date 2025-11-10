#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup.py

Configuración de instalación para el proyecto de preconvergencia refactorizado.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Leer requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().splitlines()

setup(
    name="preconvergencia-gaas",
    version="2.0.0",
    author="Investigador",
    author_email="investigador@example.com",
    description="Pipeline modular de preconvergencia DFT/PBC para GaAs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/preconvergencia-gaas",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],

    python_requires=">=3.9",

    install_requires=requirements,

    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "gpu": [
            "gpu4pyscf>=1.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.2",
        ],
    },

    entry_points={
        "console_scripts": [
            "run-preconvergence=scripts.run_preconvergence:main",
            "visualize-preconvergence=scripts.visualize_preconvergence:main",
        ],
    },

    include_package_data=True,

    zip_safe=False,
)