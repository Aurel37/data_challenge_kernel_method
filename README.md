# data_challenge_kernel_method

Data challenge for the MVA course kernel method

# Installation

You need all the classical module : numpy, scipy... as well as the two optimizations module : cvxopt and cvxpy.

The best is to create a virtual envirnonement and to install the dependencies locally with python :
    - ```python -m venv env```
    - ```source env/bin/activate```
    - ```pip install -r requirements```

# Run the code

To run the code you simply need : ```python start.py```

You can also run the main file with two more arguments :  ```python main.py -k``` if you want to test the kmeans or ```python main.py -h``` if you want to test the pca extraction after the hog extraction.

# Methods implemented

The methods used for this challenge are :
    - Kernel PCA
    - Kernel Kmeans
    - HOG
    - MultiClass kernel SVM

