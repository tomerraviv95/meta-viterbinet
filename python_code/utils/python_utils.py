import pickle as pkl
import numpy as np
import torch


def save_pkl(pkls_path: str, array: np.ndarray):
    output = open(pkls_path, 'wb')
    pkl.dump(array, output)
    output.close()


def load_pkl(pkls_path: str):
    output = open(pkls_path, 'rb')
    return pkl.load(output)


def copy_model(source_model: torch.nn.Module, dest_model: torch.nn.Module):
    """
    Copy all model parameters from source to dest
    :param source_model: model from which to copy the parameters
    :param dest_model: copy-to model
    """
    source_model_params = list(source_model.parameters())
    dest_model_params = list(dest_model.parameters())
    n = len(source_model_params)
    for i in range(0, n):
        dest_model_params[i].data[:] = source_model_params[i].data[:]
