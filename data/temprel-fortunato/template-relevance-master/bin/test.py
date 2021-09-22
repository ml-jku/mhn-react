import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed

from temprel.models import relevance
from temprel.rdkit import smiles_to_fingerprint, templates_from_smarts_list
from temprel.evaluate.diversity import diversity
from temprel.evaluate.accuracy import accuracy_by_popularity
from temprel.evaluate.topk_appl import topk_appl_recall_and_precision
from temprel.evaluate.reciprocal_rank import reciprocal_rank_by_popularity

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test a Morgan fingerprint teplate relevance network')
    parser.add_argument('--test-smiles', dest='test_smiles', default='data/processed/test.input.smiles.npy')
    parser.add_argument('--test-labels', dest='test_labels', default='data/processed/test.labels.classes.npy')
    parser.add_argument('--test-appl-labels', dest='test_appl_labels', default='data/processed/test.appl_matrix.npz')
    parser.add_argument('--train-labels', dest='train_labels', default='data/processed/train.labels.classes.npy')
    parser.add_argument('--templates', dest='templates_path')
    parser.add_argument('--topk', dest='topk', type=int, default=100)
    parser.add_argument('--fp-length', dest='fp_length', type=int, default=2048)
    parser.add_argument('--fp-radius', dest='fp_radius', type=int, default=2)
    parser.add_argument('--num-hidden', dest='num_hidden', type=int, default=1)
    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=1024)
    parser.add_argument('--num-highway', dest='num_highway', type=int, default=0)
    parser.add_argument('--activation', dest='activation', default='relu')
    parser.add_argument('--model-weights', dest='model_weights', default=None)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=512)
    parser.add_argument('--model-name', dest='model_name', default='baseline')
    parser.add_argument('--accuracy', dest='accuracy', action='store_true', default=False)
    parser.add_argument('--reciprocal-rank', dest='rr', action='store_true', default=False)
    parser.add_argument('--topk_appl', dest='topk_appl', action='store_true', default=False)
    parser.add_argument('--diversity', dest='diversity', action='store_true', default=False)
    parser.add_argument('--nproc', dest='nproc', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists('evaluation'):
        os.makedirs('evaluation')
    args = parse_arguments()
    test_smiles = np.load(args.test_smiles)
    test_labels = np.load(args.test_labels)
    train_labels = np.load(args.train_labels)
    if os.path.exists(args.test_appl_labels):
        test_appl_labels = sparse.load_npz(args.test_appl_labels)

    test_fps = Parallel(n_jobs=args.nproc, verbose=1)(
        delayed(smiles_to_fingerprint)(smi, length=args.fp_length, radius=args.fp_radius) for smi in test_smiles
    )
    test_fps = np.array(test_fps)

    templates = pd.read_json(args.templates_path)
    
    model = relevance(
        input_shape=(args.fp_length), output_shape=len(templates), num_hidden=args.num_hidden,
        hidden_size=args.hidden_size, activation=args.activation, num_highway=args.num_highway
    )
    model.load_weights(args.model_weights)

    if args.accuracy:
        acc = accuracy_by_popularity(model, test_fps, test_labels, train_labels, batch_size=args.batch_size)
        pd.DataFrame.from_dict(acc, orient='index', columns=model.metrics_names).to_json('evaluation/{}.accuracy.json'.format(args.model_name))
    
    if args.rr:
        rr = reciprocal_rank_by_popularity(model, test_fps, test_labels, train_labels, batch_size=args.batch_size)
        with open('evaluation/{}.recip_rank.json'.format(args.model_name), 'w') as f:
            json.dump(rr, f)

    if args.topk_appl:
        topk_appl_recall, topk_appl_precision = topk_appl_recall_and_precision(model, test_fps, test_appl_labels)
        with open('evaluation/{}.appl_recall.json'.format(args.model_name), 'w') as f:
            json.dump(topk_appl_recall, f)
        with open('evaluation/{}.appl_precision.json'.format(args.model_name), 'w') as f:
            json.dump(topk_appl_precision, f)

    if args.diversity:
        templates_rxn = np.array(templates_from_smarts_list(templates['reaction_smarts'], nproc=args.nproc))
        div = diversity(model, test_smiles, templates_rxn, topk=args.topk, fp_length=args.fp_length, fp_radius=args.fp_radius, nproc=args.nproc)
        np.save('evaluation/{}.diversity.npy'.format(args.model_name), div)