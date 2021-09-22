import argparse
import pandas as pd
from temprel.models import relevance
import tensorflow as tf

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a Morgan fingerprint teplate relevance network')
    parser.add_argument('--num-classes', dest='num_classes', type=int)
    parser.add_argument('--templates', dest='templates')
    parser.add_argument('--fp-length', dest='fp_length', type=int, default=2048)
    parser.add_argument('--fp-radius', dest='fp_radius', type=int, default=2)
    parser.add_argument('--num-hidden', dest='num_hidden', type=int, default=1)
    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=1024)
    parser.add_argument('--num-highway', dest='num_highway', type=int, default=0)
    parser.add_argument('--activation', dest='activation', default='relu')
    parser.add_argument('--model-weights', dest='model_weights', default=None)
    parser.add_argument('--model-name', dest='model_name', default='template-relevance-appl')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if not args.num_classes and not args.templates:
        raise ValueError('Error: --num-classes or --templates required')
    if args.num_classes:
        num_classes = args.num_classes
    else:
        templates = pd.read_json(args.templates)
        num_classes = len(templates)

    model = relevance(
        input_shape=(args.fp_length), output_shape=num_classes, num_hidden=args.num_hidden,
        hidden_size=args.hidden_size, activation=args.activation, num_highway=args.num_highway,
        compile_model=False
    )
    model.load_weights(args.model_weights)
    model.add(tf.keras.layers.Activation('softmax'))
    model.save(args.model_name, save_format='tf')
