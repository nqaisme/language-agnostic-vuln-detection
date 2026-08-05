from model import extractor, neutral_feature_builder, simple_classifier, ds_sample
from datasets import load_dataset
import VARS, numpy as np, argparse, os, torch, datetime
from tmp import tmp_dataset
from datasets import load_from_disk

def save_tensor(tensor: dict, split: str, type: str, args):
    if not args.result_dir:
        result_dir = os.path.join(os.getcwd(), 'tensors')
    else:
        result_dir = os.path.join(args.output_dir, 'tensors')

    os.makedirs(result_dir, exist_ok=True)
    split_dir = os.path.join(result_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    torch.save(tensor, os.path.join(split_dir, f'{type}.pt'))            
    print(f'embedding tensor saved successfully to {os.path.join(split_dir, f'{type}.pt')}!\n')


def build_features(dataset, args):
    cx = extractor(VARS.CB, batch_size=args.batch_size, max_length=args.max_length)
    gcx = extractor(VARS.GCB, batch_size=args.batch_size, max_length=args.max_length)
    builder = neutral_feature_builder()
    
    result = {}
    for split in ['train', 'test']:
        result[split] = {}
        for type in ['ec', 'eg', 'nf']:
            match type:
                case 'ec':
                    ebd = cx(list(dataset[split]['function']))
                case 'eg':
                    ebd = gcx(list(dataset[split]['function']))
                case 'nf':
                    ebd = builder.build(result[split]['ec'].X, result[split]['eg'].X)
            
            result[split][type] = ds_sample(
                ebd,
                np.array(dataset[split]['label']),
                type
            )
            save_tensor(ebd, split, type, args)
    return result

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--source-col',
        type=str,
        default=None,
        help='name of column which contains source code',
        required=True
    )
    
    parser.add_argument(
        '--label-col',
        type=str,
        default=None,
        help='name of column which contains label of source code',
        required=True
    )
        
    parser.add_argument(
        '--max-iter',
        type=int,
        default=1000,
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=512
    )
    
    parser.add_argument(
        '--norm',
        type=str,
        default='l2',
        choices=['l2', 'minmax']
    )
    
    parser.add_argument(
        '--fuse',
        type=str,
        default='concat',
        choices=['concat', 'average', 'weighted']
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5
    )
    
    parser.add_argument(
        '--pca-dim',
        type=int,
        required=False
    )
    
    parser.add_argument(
        '--do-train',
        action='store_true'
    )
    
    parser.add_argument(
        '--do-test',
        action='store_true'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=None,
        required=True,
    )
    
    parser.add_argument(
        '--result-dir',
        type=str,
        default=None,
        required=True
    )
    
    parser.add_argument(
        '--result-subdir',
        type=str,
        default=None,
        required=True
    )
    
    parser.add_argument(
        '--tensor-dir',
        type=str,
        defaut='tensors',
    )
    
    
    
    parser.add_argument()
    
    return parser.parse_args()

def main(args):

    # dataset = load_dataset(args.dataset, args.subset).select_columns(
    #     [args.source_col, args.label_col]
    #     ).rename_columns(
    #         {args.source_col: 'function', args.label_col: 'label'}
    #         )
    
    dataset = load_from_disk(args.dataset_dir).rename_columns(
        {
            args.source_col: 'function',
            args.label_col: 'label'
        }
    )
    
    
    subdir_name = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    args.result_dir = os.path.join(args.result_dir, subdir_name)

    
    if args.do_train:
        features = build_features(dataset, args)
        classifier = simple_classifier(max_iter=args.max_iter, random_state=args.random_state)
        results = {}
        
        for type in ['ec', 'eg', 'nf']:
            results[type] = classifier.train(features['train'][type], **{"output_dir": args.result_dir, 'dataset': os.path.basename(args.dataset_dir)})
    
    if args.do_test:
        weight_paths = {
            'ec': os.path.join(args.result_dir, args.result_subdir, 'ec_weights.pkl'),
            'eg': os.path.join(args.result_dir, args.result_subdir, 'eg_weights.pkl'),
            'nf': os.path.join(args.result_dir, args.result_subdir, 'nf_weights.pkl'),            
        }
        tensor_paths = {
            'ec': os.path.join(args.result_dir, args.result_subdir, args.tensor_dir, '/test/ec.pt'),
            'eg': os.path.join(args.result_dir, args.result_subdir, args.tensor_dir, '/test/eg.pt'),
            'nf': os.path.join(args.result_dir, args.result_subdir, args.tensor_dir, '/test/nf.pt'),
        }
        
        y_true = np.array(dataset['test']['label'])
        
        for type, weight_file in weight_paths.items():
            sample = ds_sample(
                torch.load(tensor_paths[type]),
                y_true,
                type
            )
            print(f"{type} embedding result:\t{simple_classifier.evaluate(weight_file, sample)}")


if __name__ == '__main__':
    args = parse_args()
    main(args)