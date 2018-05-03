# Source code for producing the results and figures

The code is divided between Python modules in `mypackage` and Jupyter notebooks
in `notebooks`. The modules implement the methodology and code that is reused
in different applications. The notebooks perform the data analysis and processing and
generate the figures for the paper.

The `Makefile` automates all processes related to executing code.
Run the following to perform all actions from building the software to
generating the final figures:

    make all

*Note: This is an incredibly expensive call to make. It will take a while.*

## Python package

*Describe the package here: what it does, what functions it has, etc*.


## Building, testing, and linting

Use the `Makefile` to build, test, and lint the software:

* Build and install:

        make build

* Run the static checks using flake8 and pylint:

        make check

* Run the tests in `tests` and doctests in docstrings:

        make test

* Calculate the test coverage of the main Python code (not including the
  notebooks):

        make coverage


## Generating results and figures

The Jupyter notebooks produce most of the results and figures. The `Makefile`
can execute the notebooks to generate these outputs. This is better than
executing the notebooks by hand because it ensures that cells are run
sequentially in a way that can be reproduced.

* Generate all results files specified in the `Makefile`:

        make results

* Create all figure files specified in the `Makefile`:

        make figures


## Notebooks

* [Full-Analysis.ipynb](http://nbviewer.jupyter.org/github/danielnewman09/Catheter-Ablation/blob/master/code/notebooks/Full-Analysis.ipynb):
  This notebook is an experimental sandbox for my modeling and control methods. I don't expect it to be exceptionally readable.
* [Create-Model.ipynb](http://nbviewer.jupyter.org/github/danielnewman09/Catheter-Ablation/blob/master/code/notebooks/Create-Model.ipynb):
  Derive the full nonlinear model and compare it to a linearized version. The functions here are incorporated into the package to allow for easy replication across different scripts.
* [Control-Analysis.ipynb](http://nbviewer.jupyter.org/github/danielnewman09/Catheter-Ablation/blob/master/code/notebooks/Control-Analysis.ipynb):
  Create the model predictive controller and analyze its performance relative to PD control.
