Experimental codes for AISTATS 2018 paper "Efficient and principled score estimation with Nyström kernel exponential families" by Dougal Sutherland, Heiko Strathmann, Michael Arbel, and Arthur Gretton, https://arxiv.org/abs/1705.08360.

See notebooks/demo.ipynb for how to use the estimator(s), and how to replicate experimental results.

Dependencies (some are optional, see demo notebook):
* numpy, scipy, matplotlib
* Shogun, http://shogun.ml/, more specifically the code in the feature branch https://github.com/karlnapf/shogun/tree/feature/kernel_exp_family, compiled with the Python interface. We are working on pushing this into the main branch of Shogun, so that it can be installed using `conda install -c conda-forge shogun`.
* the Python package https://github.com/karlnapf/kernel_exp_family
* tensorflow

For the Python packages (given that you have downloaded them) and Shogun (given that you have compiled or installed it), this could be achieved with something like

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/libshogun.so
export PYTHONPATH=$PYTHONPATH:/path/to/shogun.py
export PYTHONPATH=$PYTHONPATH:/path/to/nystrom-kexpfam
export PYTHONPATH=$PYTHONPATH:/path/to/kernel_exp_family
```
