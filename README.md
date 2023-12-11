# AMBER2

## Installation
Environment setup for PyTorch, TF2 and Jax:
```
conda create -n amber2-dev -c conda-forge -c anaconda pytorch jax tensorflow keras pytorch-lightning ipython notebook
conda activate amber2-dev
```

### Installing with PyTorch/Lightning
```{bash}
conda create -n amber-torch -c conda-forge pytorch scikit-learn numpy scipy matplotlib seaborn tqdm h5py
conda activate amber-torch
pip install pytorch-lightning torchmetrics
# pip install amber-automl # wait for production version
git clone git@github.com:zhanglab-aim/AMBER2.git
cd AMBER2
python setup.py develop
# if you plan to run tests
pip install pytest coverage parameterized expecttest hypothesis
```

### Installing with Tensorflow 2
```{bash}
conda create -n amber-tf2 -c conda-forge tensorflow-gpu scikit-learn seaborn
# if you are on MacOS, or don't have CUDA-enabled GPU, replace tensorflow-gpu with tensorflow
conda activate amber-tf2
# pip install amber-automl # wait for production version
git clone git@github.com:zhanglab-aim/AMBER2.git
cd AMBER2
python setup.py develop
# if you plan to run tests
pip install pytest coverage parameterized pydot graphviz
```
