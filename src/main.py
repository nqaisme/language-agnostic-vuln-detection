from model import extractor, neutral_feature_builder, simple_classifier
from datasets import load_dataset
import VARS, numpy as np


def build_features(dataset):
    cx = extractor(VARS.CB)
    gcx = extractor(VARS.GCB)
    builder = neutral_feature_builder()
    
    train_feat = {
        'ec': cx(list(dataset['train']['function'])),
        'eg': gcx(list(dataset['train']['function']))
    }
    train_feat['nf'] = builder.build(train_feat['ec'], train_feat['eg'])
    
    
    test_feat = {
        'ec': cx(list(dataset['test']['function'])),
        'eg': gcx(list(dataset['test']['function']))
    }
    test_feat['nf'] = builder.build(test_feat['ec'], test_feat['eg'])

    return {
        'x_train': train_feat,
        'x_test': test_feat,
        'y_train': np.array(dataset['train']['label']),
        'y_test': np.array(dataset['test']['label'])
    }


def main():
    primevul = load_dataset('colin/PrimeVul').select_columns(['func', 'target']).rename_columns({'func': 'function', 'target': 'label'})
    
    features = build_features(primevul)

    classifier = simple_classifier()

    f = classifier.train(features['x_train'], features['y_train'], **{'dataset': 'primevul'})

    res = simple_classifier.evaluate(f, features['x_test'], features['y_test'])
    
    print(f'PrimeVul: {res}')
    
    
if __name__ == '__main__':
    main()