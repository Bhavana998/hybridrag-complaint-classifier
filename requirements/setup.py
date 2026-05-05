from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/base.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="hybridrag-complaint-classifier",
    version="1.0.0",
    author="Your Name",
    description="Hybrid RAG system for customer complaint classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hybridrag=run:main",
        ],
    },
)