import os
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from temprel.models import applicability
from temprel.data.loaders import fingerprint_training_dataset
import tensorflow as tf
import sklearn

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a Morgan fingerprint teplate relevance network')
    parser.add_argument('--train-smiles', dest='train_smiles', default='data/processed/train.input.smiles.npy')
    parser.add_argument('--valid-smiles', dest='valid_smiles', default='data/processed/valid.input.smiles.npy')
    parser.add_argument('--train-labels', dest='train_labels', default='data/processed/train.appl_matrix.npz')
    parser.add_argument('--valid-labels', dest='valid_labels', default='data/processed/valid.appl_matrix.npz')
    parser.add_argument('--no-validation', dest='no_validation', action='store_true', default=False)
    parser.add_argument('--fp-length', dest='fp_length', type=int, default=2048)
    parser.add_argument('--fp-radius', dest='fp_radius', type=int, default=2)
    parser.add_argument('--precompute-fps', dest='precompute_fps', action='store_true', default=True)
    parser.add_argument('--weight-classes', dest='weight_classes', action='store_true', default=True)
    parser.add_argument('--num-hidden', dest='num_hidden', type=int, default=1)
    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=1024)
    parser.add_argument('--num-highway', dest='num_highway', type=int, default=0)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.2)
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--activation', dest='activation', default='relu')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('--clipnorm', dest='clipnorm', action='store_true', default=None)
    parser.add_argument('--epochs', dest='epochs', type=int, default=25)
    parser.add_argument('--early-stopping', dest='early_stopping', type=int, default=3)
    parser.add_argument('--model-name', dest='model_name', default='template-relevance-appl')
    parser.add_argument('--nproc', dest='nproc', type=int, default=0)
    return parser.parse_args()

def try_read_npz(filename):
    if not os.path.exists(filename):
        raise ValueError('File does not exist: {}'.format(filename))
    return sparse.load_npz(filename)

def try_read_npy(filename):
    if not os.path.exists(filename):
        raise ValueError('File does not exist: {}'.format(filename))
    return np.load(filename)

def shuffle_arrays(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == '__main__':
    args = parse_arguments()
    if not args.train_smiles or not args.train_labels:
        raise ValueError('Error: training data (--train-smiles and --train-labels) required')
    train_smiles = try_read_npy(args.train_smiles)
    train_labels = try_read_npz(args.train_labels)
    train_smiles, train_labels = shuffle_arrays(train_smiles, train_labels)
    if not args.no_validation:
        valid_smiles = try_read_npy(args.valid_smiles)
        valid_labels = try_read_npz(args.valid_labels)
        valid_smiles, valid_labels = shuffle_arrays(valid_smiles, valid_labels)

    num_classes = train_labels.shape[1]

    if args.weight_classes:
        template_example_counts = train_labels.sum(axis=0).A.reshape(-1)
        template_example_counts[np.argwhere(template_example_counts==0).reshape(-1)] = 1
        template_class_weights = template_example_counts.sum()/template_example_counts.shape[0]/template_example_counts
    else:
        template_class_weights = {n: 1. for n in range(train_labels.shape[1])}

    train_ds = fingerprint_training_dataset(
        train_smiles, train_labels, batch_size=args.batch_size, train=True,
        fp_length=args.fp_length, fp_radius=args.fp_radius, nproc=40,
        sparse_labels=True, cache=False, precompute=args.precompute_fps
    )
    train_steps = np.ceil(len(train_smiles)/args.batch_size).astype(int)

    if not args.no_validation:
        valid_ds = fingerprint_training_dataset(
            valid_smiles, valid_labels, batch_size=args.batch_size, train=False,
            fp_length=args.fp_length, fp_radius=args.fp_radius, nproc=40,
            sparse_labels=True, cache=False, precompute=args.precompute_fps
        )
        valid_steps = np.ceil(len(valid_smiles)/args.batch_size).astype(int)
    else:
        valid_ds = None
        valid_steps = None

    model = applicability(
        input_shape=(args.fp_length,),
        output_shape=num_classes,
        num_hidden=args.num_hidden,
        hidden_size=args.hidden_size,
        num_highway=args.num_highway,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        activation=args.activation,
        clipnorm=args.clipnorm
    )

    if not os.path.exists('training'):
        os.makedirs('training')
    model_output = 'training/{}-weights.hdf5'.format(args.model_name)
    history_output = 'training/{}-history.json'.format(args.model_name)

    callbacks = []
    if args.early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=args.early_stopping,
                restore_best_weights=True
            )
        )
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            model_output, monitor='val_loss', save_weights_only=True
        )
    )

    if args.nproc:
        multiproc = True
        nproc = args.nproc
    else:
        multiproc = False
        nproc = None

    history = model.fit(
        train_ds, epochs=args.epochs, steps_per_epoch=train_steps,
        validation_data=valid_ds, validation_steps=valid_steps,
        callbacks=callbacks, class_weight=template_class_weights,
        use_multiprocessing=multiproc, workers=nproc
    )

    pd.DataFrame(history.history).to_json(history_output)