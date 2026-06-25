<div align="center" style="background-color: white;">
<a href="https://www.transportforthenorth.com/">
<img src="https://www.transportforthenorth.com/logo.svg" alt="Transport for the North logo">
</a>
</div>

<h1 align="center">CAF.mat</h1>

<p align="center">
<a href="https://transport-for-the-north.github.io/CAF-Handbook/python_tools/framework.html">
  <img alt="CAF Status - Pre-Alpha" src="https://img.shields.io/badge/CAF%20Status-Pre--Alpha-orange">
</a>
</p>
<p align="center">
<a href="https://pypi.org/project/caf.mat/">
  <img alt="Supported Python versions" src="https://img.shields.io/pypi/pyversions/caf.mat.svg?style=flat-square">
</a>
<a href="https://pypi.org/project/caf.mat/">
  <img alt="Latest release" src="https://img.shields.io/github/release/transport-for-the-north/caf.mat.svg?style=flat-square&maxAge=86400">
</a>
<a href="https://anaconda.org/conda-forge/caf.mat">
  <img alt="Conda" src="https://img.shields.io/conda/v/conda-forge/caf.mat?style=flat-square&logo=condaforge">
</a>
</p>
<p align="center">
<a href="https://github.com/transport-for-the-north/caf.mat/actions?query=event%3Apush">
  <img alt="Testing Badge" src="https://img.shields.io/github/actions/workflow/status/transport-for-the-north/caf.mat/tests.yml?style=flat-square&logo=GitHub&label=Tests">
</a>
<a href="https://app.codecov.io/gh/transport-for-the-north/caf.mat">
  <img alt="Coverage" src="https://img.shields.io/codecov/c/github/transport-for-the-north/caf.mat.svg?branch=main&style=flat-square&logo=CodeCov">
</a>
<a href='https://cafmat.readthedocs.io/en/stable/'>
  <img alt='Documentation Status' src="https://img.shields.io/readthedocs/cafmat?style=flat-square&logo=readthedocs">
</a>
</p>

> [!WARNING]
> This package is in an early stage of development so features may change or be removed.
> If using this package it is recommended to set a specific version and check before
> upgrading to a new version.

CAF.mat contains functionality for reading, writing and performing calculations with transport
modelling matrices. The aim is to provide a consistent method for interacting with all types of
transport matrices from Python, with some additional front-end features for more common matrix
operations.

> [!TIP]
> For more detailed information including a user guide, tutorials and API reference see the full
> [caf.mat documentation](https://cafmat.readthedocs.io/en/stable/)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [What does it do?](#what-does-it-do)
  - [Main Features](#main-features)
    - [Work-in-Progress](#work-in-progress)
  - [Who is it for?](#who-is-it-for)
- [Where to get it](#where-to-get-it)
  - [Installation from GitHub](#installation-from-github)
- [Usage](#usage)
  - [Command Line](#command-line)
- [Documentation](#documentation)
- [What is CAF?](#what-is-caf)
- [Contribution](#contribution)
- [Contact Us](#contact-us)

## Overview

### What does it do?

CAF.mat is primarily a Python package for interacting with transport modelling matrices in various
file formats. The package provides a set of classes for interacting with matrices and ties into
some other CAF packages, namely [caf.base](https://cafbase.readthedocs.io/en/stable/).

The package also provides a standalone CLI tool for some more common matrix operations.

### Main Features

- **Matrices Classes** - custom Python classes to handle groups of matrices, builds on top of
  [caf.base's](https://cafbase.readthedocs.io/en/stable/)
  [`Segmentation`](https://cafbase.readthedocs.io/en/stable/_autosummary/caf.base.Segmentation.html)
  and [`ZoningSystem`](https://cafbase.readthedocs.io/en/stable/_autosummary/caf.base.ZoningSystem.html)
  functionality and provides access to individual matrices as
  [pandas DataFrames](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
- **Various File Formats** - read, write and convert between various matrix file formats from
  different transport modelling software.
  - OpenMatrix (OMX)
  - SATURN's UFM
  - CUBE's (MAT)
- **Any Matrix Data** - can handle any type of matrix data e.g. travel demand and transport costs.
- **Tour Modelling Conversions** - functions to convert between OD and PA based matrices, as used
  in tour based transport demand modelling.

#### Work-in-Progress

- **UFM Comparisons** - functionality to compare different UFM files (see #17).
- **Matrix Adjustments** - adjust a matrix to a target matrix at a specified sector level (see #18).

> [!WARNING]
> These features are work-in-progress and are not available in a released version of caf.mat, to
> access these features a specific branch of caf.mat should be installed, see [Installation from GitHub](#installation-from-github).

### Who is it for?

- **Target audience:** Transport Modellers
- **CAF Analytical Stage:** Modelling, Appraisal

![CAF Analytical Process Diagram](https://github.com/Transport-for-the-North/.github/blob/21a428e81880639839e221940881572cdee24d5a/profile/ProcessDiagram.png?raw=true)

For more details on CAF Analytical Stages see the [description within TfN's GitHub homepage](https://github.com/Transport-for-the-North)

## Where to get it

The latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/caf.mat) and on [Conda](https://anaconda.org/conda-forge/caf.mat).

```sh
conda install -c conda-forge caf.mat
```

```sh
pip install caf.mat
```

> [!TIP]
>
> - See the [Quick Start Guide](https://cafmat.readthedocs.io/en/stable/start.html#quick-start) for more detailed instructions.
> - See the [requirements.txt](requirements.txt) for the full list of package dependencies.

### Installation from GitHub

> [!WARNING]
> Unreleased GitHub versions should **not** be considered stable.

The latest, unreleased, version can be installed directly from GitHub using:

```sh
pip install "git+https://github.com/transport-for-the-north/caf.mat"
```

> [!TIP]
> `pip install` can install a specific tag, or branch, using `@{tag-name}`
> after the git URL.

## Usage

caf.mat provides and Command-line (CLI) and graphical interface (GUI) to use many of it's
features without the need to write any Python code, see the [Tool Usage section](https://cafmat.readthedocs.io/en/stable/usage/index.html)
of the user guide for more details.

### Command Line

The tool can be run from command line, with the command:

```sh
caf.mat
```

See [Command-Line Interface (User Guide)](https://cafmat.readthedocs.io/en/stable/usage/cli.html)
for full explanations of the parameters.

## Documentation

The code documentation is hosted at <https://cafmat.readthedocs.io/en/stable/>.

## What is CAF?

This tool is part of TfN's [Common Analytical Framework (CAF)](https://github.com/Transport-for-the-North).
CAF is Transport for the North's structured suite of analytical tools designed to support transport
modelling, appraisal, and strategic decision-making.

More information on CAF and details on other CAF tools can be found on [TfN's GitHub Homepage](https://github.com/Transport-for-the-North).

## Contribution

We encourage use of, and contributions to, the repositories within this organisation, licenses are provided within
the repositories and the [organisation contribution guide](https://github.com/Transport-for-the-North/.github/blob/main/CONTRIBUTING.rst)
provides details for contributions.

---

## Contact Us

For further information about using this tool or CAF tools in your projects and work contact Transport for the North - <TfNOffer@transportforthenorth.com>

---

[Go to Top](#table-of-contents)
