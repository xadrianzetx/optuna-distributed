import os

from setuptools import find_packages
from setuptools import setup


def get_version() -> str:
    with open(os.path.join("optuna_distributed", "__init__.py")) as file:
        for line in file.readlines():
            if line.startswith("__version__"):
                version = line.strip().split(" = ")[1]
                return version.replace('"', "")

        else:
            raise RuntimeError("Didn't manage to find version string.")


def get_long_description() -> str:
    with open("README.md") as file:
        return file.read()


setup(
    name="optuna_distributed",
    version=get_version(),
    author="Adrian Zuber",
    author_email="xadrianzetx@gmail.com",
    description="Distributed hyperparameter optimization made easy",
    long_description_content_type="text/markdown",
    long_description=get_long_description(),
    url="https://github.com/xadrianzetx/optuna-distributed",
    project_urls={
        "Source": "https://github.com/xadrianzetx/optuna-distributed",
        "Bug Tracker": "https://github.com/xadrianzetx/optuna-distributed/issues",
    },
    packages=find_packages(exclude=["tests"]),
    lassifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.7",
    # TODO(xadrianzetx) Unpin Optuna after V3 release.
    # TODO(xadrianzetx) Remove typing-extensions when Python 3.7 is out of support.
    # We are using typing-extensions are used for typing.Literal.
    install_requires=["optuna==3.0.0rc0", "dask[distributed]", "typing-extensions"],
    extras_require={
        "dev": ["black", "isort", "flake8", "mypy", "pandas", "pandas-stubs"],
        "test": ["pytest"],
    },
)
