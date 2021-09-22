import os
import argparse
import numpy as np
import pandas as pd
from temprel.models import relevance
from temprel.data.loaders import fingerprint_training_dataset
import tensorflow as tf
import sklearn

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a Morgan fingerprint teplate relevance network')
    parser.add_argument('--train-smiles', dest='train_smiles', default='data/processed/train.input.smiles.npy')
    parser.add_argument('--valid-smiles', dest='valid_smiles', default='data/processed/valid.input.smiles.npy')
    parser.add_argument('--train-labels', dest='train_labels', default='data/processed/train.labels.classes.npy')
    parser.add_argument('--valid-labels', dest='valid_labels', default='data/processed/valid.labels.classes.npy')
    parser.add_argument('--no-validation', dest='no_validation', action='store_true', default=False)
    parser.add_argument('--num-classes', dest='num_classes', type=int)
    parser.add_argument('--templates', dest='templates', default='data/processed/retro.templates.json.gz')
    parser.add_argument('--fp-length', dest='fp_length', type=int, default=2048)
    parser.add_argument('--fp-radius', dest='fp_radius', type=int, default=2)
    parser.add_argument('--precompute-fps', dest='precompute_fps', action='store_true', default=True)
    parser.add_argument('--pretrain-weights', dest='pretrain_weights', default=None)
    parser.add_argument('--weight-classes', dest='weight_classes', action='store_true', default=True)
    parser.add_argument('--num-hidden', dest='num_hidden', type=int, default=1)
    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=1024)
    parser.add_argument('--num-highway', dest='num_highway', type=int, default=0)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.2)
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--activation', dest='activation', default='relu')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=512)
    parser.add_argument('--clipnorm', dest='clipnorm', action='store_true', default=None)
    parser.add_argument('--epochs', dest='epochs', type=int, default=25)
    parser.add_argument('--early-stopping', dest='early_stopping', type=int, default=3)
    parser.add_argument('--model-name', dest='model_name', default='template-relevance')
    parser.add_argument('--nproc', dest='nproc', type=int, default=0)
    return parser.parse_args()

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
    train_labels = try_read_npy(args.train_labels)
    train_smiles, train_labels = shuffle_arrays(train_smiles, train_labels)
    if not args.no_validation:
        valid_smiles = try_read_npy(args.valid_smiles)
        valid_labels = try_read_npy(args.valid_labels)
        valid_smiles, valid_labels = shuffle_arrays(valid_smiles, valid_labels)
    if not args.num_classes and not args.templates:
        raise ValueError('Error: --num-classes or --templates required')
    if args.num_classes:
        num_classes = args.num_classes
    else:
        templates = pd.read_json(args.templates)
        num_classes = len(templates)

    train_ds = fingerprint_training_dataset(
        train_smiles, train_labels, batch_size=args.batch_size, train=True,
        fp_length=args.fp_length, fp_radius=args.fp_radius, nproc=40, precompute=args.precompute_fps
    )
    train_steps = np.ceil(len(train_smiles)/args.batch_size).astype(int)

    if not args.no_validation:
        valid_ds = fingerprint_training_dataset(
            valid_smiles, valid_labels, batch_size=args.batch_size, train=False,
            fp_length=args.fp_length, fp_radius=args.fp_radius, nproc=40, precompute=args.precompute_fps
        )
        valid_steps = np.ceil(len(valid_smiles)/args.batch_size).astype(int)
    else:
        valid_ds = None
        valid_steps = None

    model = relevance(
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
    if args.pretrain_weights:
        model.load_weights(args.pretrain_weights)

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

    if args.weight_classes:
        class_weight = sklearn.utils.class_weight.compute_class_weight(
            'balanced', np.unique(train_labels), train_labels
        )
    else:
        class_weight = sklearn.utils.class_weight.compute_class_weight(
            None, np.unique(train_labels), train_labels
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
        callbacks=callbacks, class_weight=class_weight,
        use_multiprocessing=multiproc, workers=nproc
    )

    pd.DataFrame(history.history).to_json(history_output)