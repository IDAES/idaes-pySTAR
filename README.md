# pySTAR Toolbox
Python Symbolic regression Through Algebraic Representations (pySTAR) provides tools for generating mathematical (symbolic) models that best fit some input data. The pySTAR toolbox allows for the definition of the form of the surrogate model and the regression parameter values simultaneously.

# Reference
If you use this code please cite: "Sarwar, O. 2022, Algorithms for Interpretable High-Dimensional Regression, Carnegie Mellon University."

# Installation instructions

Ensure that Anaconda is installed, and follow the steps below:

1. Open Anaconda Prompt
2. Clone the repository on your computer
```bash
git clone https://github.com/IDAES/idaes-pySTAR.git
```
3. Create a new `conda` environment called `pystar` by running
```bash
conda create -n pystar python==3.13
conda activate pystar
```
4. Navigate to the repository
```bash
cd path/to/cloned/idaes-pySTAR/repository
```
5. Install the package by running
```bash
pip install -r requirements-dev.txt
```

# Configure Logger

Logger can be configured by adding the following lines at the top of your script.

```python
from pystar import setup_logger
setup_logger()
```

To save the log to a file, pass the `log_file` argument: `setup_logger(log_file="path-to-log-file.log")`

# Funding Acknowledgements

This work was conducted as part of the Institute for the Design of Advanced Energy Systems (IDAES) with support through the Crosscutting Research Program within the U.S. Department of Energy’s Office of Fossil Energy and Carbon Management (FECM).

# Citation

If you use this code please cite: "Sarwar, O. 2022, Algorithms for Interpretable High-Dimensional Regression, Carnegie Mellon University."

