from model import extractor, neutral_feature_builder, simple_classifier, ds_sample
from datasets import load_dataset
import VARS, numpy as np


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
                builder.build(*(list(feats['train'].values())))
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
                builder.build(*(list(feats['test'].values())))
            )
        }
    }


def main():
    primevul = load_dataset('colin/PrimeVul', 'paired').select_columns(['func', 'target']).rename_columns({'func': 'function', 'target': 'label'})
    
    features = build_features(primevul)

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
    main()