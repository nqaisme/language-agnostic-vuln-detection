from model import extractor, neutral_feature_builder
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import VARS, numpy as np



def evaluate(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> dict:
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }



ds_train = load_dataset('colin/PrimeVul', 'paired', split='train').select_columns(['target', 'func'])
ds_test = load_dataset('colin/PrimeVul', 'paired', split='test').select_columns(['target', 'func'])




cx = extractor(model_name=VARS.CB, max_length=VARS.MAX_LENGTH, batch_size=VARS.BATCH_SIZE)
gcx = extractor(model_name=VARS.GCB, max_length=VARS.MAX_LENGTH, batch_size=VARS.BATCH_SIZE)


ecs = cx(list(ds_train['func']))
egcs = gcx(list(ds_train['func']))
nf = neutral_feature_builder().build(ecs, egcs)

feat_tests = {
    'ec': cx(list(ds_test['func'])),
    'eg': gcx(list(ds_test['func'])),
}

feat_tests['nf'] = neutral_feature_builder().build(
    feat_tests.get('ec'),
    feat_tests.get('eg')
)


lr1 = LogisticRegression(max_iter=VARS.MAX_ITER)
lr2 = LogisticRegression(max_iter=VARS.MAX_ITER)
lr3 = LogisticRegression(max_iter=VARS.MAX_ITER)


y_train, y_test = np.array(ds_train['target']), np.array(ds_test['target'])
lr1.fit(ecs, y_train)
lr2.fit(egcs, y_train)
lr3.fit(nf, y_train)


res = {
    'ec': evaluate(y_test, lr1.predict(feat_tests['ec'])),
    'eg': evaluate(y_test, lr2.predict(feat_tests['eg'])),
    'nf': evaluate(y_test, lr3.predict(feat_tests['nf']))
}


for k, v in res.items():
    print(f'{k}: {v}\n\n')