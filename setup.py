from setuptools import setup, find_packages

setup(
    name="adasyn",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.7",
)