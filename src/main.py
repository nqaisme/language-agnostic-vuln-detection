from model import extractor, neutral_feature_builder, simple_classifier, ds_sample
from datasets import load_dataset
import VARS, numpy as np, argparse


def build_features(dataset):
    cx = extractor(VARS.CB)
    gcx = extractor(VARS.GCB)
    builder = neutral_feature_builder()
    
    train_codes = list(dataset['train']['function'])
    test_codes = list(dataset['test']['function'])
    y_train, y_test = np.array(dataset['train']['label']), np.array(dataset['test']['label'])

    feats = {
        'train': {
            'ec': cx(train_codes),
            'eg': gcx(train_codes)
        },
        'test': {
            'ec': cx(test_codes),
            'eg': gcx(test_codes)
        }
    }
    
    
    return {
        'train': {
            'ec': ds_sample(
                feats['train']['ec'],
                y_train,
                'ec'
            ),
            'eg': ds_sample(
                feats['train']['eg'],
                y_train,
                'eg'
            ),
            'nf': ds_sample(
                builder.build(*(list(feats['train'].values()))),
                y_train,
                'nf'
            )
        },
        'test': {
            'ec': ds_sample(
                feats['test']['ec'],
                y_test,
                'ec'
            ),
            'eg': ds_sample(
                feats['test']['eg'],
                y_test,
                'eg'
            ),
            'nf': ds_sample(
                builder.build(*(list(feats['test'].values()))),
                y_test,
                'nf'
            )
        }
    }

def parse_args():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='colin/PrimeVul',
        help='huggingface dataset'
    )
    
    parser.add_argument(
        '--subset',
        type=str,
        help='subset of the given dataset'
    )
    
    parser.add_argument(
        '--source-col',
        type=str,
        default='func',
        help='name of column which contains source code'
    )

    parser.add_argument(
        '--label-col',
        type=str,
        default='target',
        help='name of column which contains label of source code'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/content/drive/MyDrive'
    )
    
    return parser.parse_args()

def main(args):
    datset = load_dataset(args.dataset, args.subset).select_columns(
        [args.source_col, args.label_col]
        ).rename_columns(
            {args.source_col: 'function', args.label_col: 'label'}
            )
    
    features = build_features(datset)

    classifier = simple_classifier()

    '''
    train phase
    '''
    results = {}
    for type in ['ec', 'eg', 'nf']:
        results[type] = classifier.train(features['train'][type])
    
    for type in ['ec', 'eg', 'nf']:
        print(simple_classifier.evaluate(results[type], features['test'][type]))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)