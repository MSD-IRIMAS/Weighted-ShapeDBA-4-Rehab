import numpy as np
import os

def normalize_skeletons(X, min_X=None, max_X=None, min_Y=None, max_Y=None, min_Z=None, max_Z=None):

    n_X = np.zeros(shape=X.shape)

    if min_X is None:
        min_X = np.min(X[:,:,:,0])
    
    if max_X is None:
        max_X = np.max(X[:,:,:,0])

    if min_Y is None:
        min_Y = np.min(X[:,:,:,1])
    
    if max_Y is None:
        max_Y = np.max(X[:,:,:,1])

    if min_Z is None:
        min_Z = np.min(X[:,:,:,2])
    
    if max_Z is None:
        max_Z = np.max(X[:,:,:,2])

    n_X[:,:,:,0] = (X[:,:,:,0] - min_X) / (1.0 * (max_X - min_X))
    n_X[:,:,:,1] = (X[:,:,:,1] - min_Y) / (1.0 * (max_Y - min_Y))
    n_X[:,:,:,2] = (X[:,:,:,2] - min_Z) / (1.0 * (max_Z - min_Z))

    return n_X, min_X, max_X, min_Y, max_Y, min_Z, max_Z

def create_directory(directory_path):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)