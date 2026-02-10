"""
Setup script for xiao_net_ver_2 package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="xiao_net_ver_2",
    version="2.0.0",
    author="Hy-X",
    description="U-Net architecture for seismic phase detection and catalog curation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hy-X/xiao_net_ver_2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'xiao-train=train:main',
            'xiao-infer=inference:main',
        ],
    },
)
