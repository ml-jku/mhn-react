import time
import argparse
import pandas as pd
from temprel.templates.extract import templates_from_reactions, process_for_training, process_for_askcos

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process reaction smiles for template relevance training with ASKCOS')
    parser.add_argument('--reactions', dest='reactions', default='data/raw/reactions.json.gz')
    parser.add_argument('--nproc', dest='nproc', type=int, default=1)
    parser.add_argument('--output-prefix', dest='output_prefix', default='data/processed/')
    parser.add_argument('--calc-split', dest='calc_split', default='stratified')
    parser.add_argument('--template-set-name', dest='template_set_name', default='uspto_50k')
    return parser.parse_args()

def print_time(task_name, t0):
    new_t0 = time.time()
    print('elapsed {}: {}'.format(task_name, new_t0-t0))
    return new_t0


if __name__ == '__main__':
    args = parse_arguments()
    t0 = time.time()
    reactions = pd.read_json(args.reactions)
    t0 = print_time('read', t0)
    templates = templates_from_reactions(reactions, nproc=args.nproc)
    t0 = print_time('extract', t0)
    process_for_training(templates, output_prefix=args.output_prefix, calc_split=args.calc_split)
    t0 = print_time('featurize', t0)
    process_for_askcos(templates, template_set_name=args.template_set_name, output_prefix=args.output_prefix)
    t0 = print_time('askcos_process', t0)