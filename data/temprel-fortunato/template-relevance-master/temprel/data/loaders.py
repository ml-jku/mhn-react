import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from ..rdkit import smiles_to_fingerprint

def fingerprint_training_dataset(
    smiles, labels, batch_size=256, train=True,
    fp_length=2048, fp_radius=2, fp_use_features=False, fp_use_chirality=True,
    sparse_labels=False, shuffle_buffer=1024, nproc=8, cache=True, precompute=False
):
    smiles_ds = fingerprint_dataset_from_smiles(smiles, fp_length, fp_radius, fp_use_features, fp_use_chirality, nproc, precompute)
    labels_ds = labels_dataset(labels, sparse_labels)
    ds = tf.data.Dataset.zip((smiles_ds, labels_ds))
    ds = ds.shuffle(shuffle_buffer).batch(batch_size)
    if train:
        ds = ds.repeat()
    if cache:
        ds = ds.cache()
    ds = ds.prefetch(buffer_size=batch_size*3)
    return ds

def fingerprint_dataset_from_smiles(smiles, length, radius, useFeatures, useChirality, nproc=8, precompute=False):
    def smiles_tensor_to_fp(smi, length, radius, useFeatures, useChirality):
        smi = smi.numpy().decode('utf-8')
        length = int(length.numpy())
        radius = int(radius.numpy())
        useFeatures = bool(useFeatures.numpy())
        useChirality = bool(useChirality.numpy())
        fp_bit = smiles_to_fingerprint(smi, length, radius, useFeatures, useChirality)
        return np.array(fp_bit)
    def parse_smiles(smi):
        output = tf.py_function(
            smiles_tensor_to_fp, 
            inp=[smi, length, radius, useFeatures, useChirality], 
            Tout=tf.float32
        )
        output.set_shape((length,))
        return output
    if not precompute:
        ds = tf.data.Dataset.from_tensor_slices(smiles)
        ds = ds.map(map_func=parse_smiles, num_parallel_calls=nproc)
    else:
        fps = Parallel(n_jobs=nproc, verbose=1)(
            delayed(smiles_to_fingerprint)(smi, length, radius, useFeatures, useChirality) for smi in smiles
        )
        fps = np.array(fps)
        ds = tf.data.Dataset.from_tensor_slices(fps)
    return ds

def labels_dataset(labels, sparse=False):
    if not sparse:
        return tf.data.Dataset.from_tensor_slices(labels)
    coo = labels.tocoo()
    indices = np.array([coo.row, coo.col]).T
    labels = tf.SparseTensor(indices, coo.data, coo.shape)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    labels_ds = labels_ds.map(map_func=tf.sparse.to_dense)
    return labels_ds