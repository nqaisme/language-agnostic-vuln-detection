from model import extractor, neutral_feature_builder, simple_classifier, ds_sample
from datasets import load_dataset
import VARS, numpy as np, argparse, os, torch, datetime
from tmp import tmp_dataset
from datasets import load_from_disk

def save_tensor(tensor: dict, split: str, type: str, args):
    if not args.output_dir:
        output_dir = os.path.join(os.getcwd(), 'tensors')
    else:
        output_dir = os.path.join(args.output_dir, 'tensors')

    os.makedirs(output_dir, exist_ok=True)
        
    
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    torch.save(tensor, os.path.join(split_dir, f'{type}.pt'))
            

    print(f'Embedding tensor saved successfully to {os.path.join(split_dir, f'{type}.pt')}!\n')
    
        

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
        '--dataset-path',
        default=None,
        type=str,
        help='path to the dataset directory, consists of train & test set',
        required=True
    )
    
    parser.add_argument(
        '--weights-path',
        default=None,
        type=str,
        help='path to model\' weights directory',
        required=True
    )
    
    parser.add_argument(
        '--features-path',
        default=None,
        type=str,
        help='path to embedding tensors directory',
        required=True
    )
    
    
    parser.add_argument(
        '--source-col',
        type=str,
        default='func',
        help='name of column which contains source code',
        required=True
    )

    parser.add_argument(
        '--label-col',
        type=str,
        default='target',
        help='name of column which contains label of source code',
        required=True
    )
    
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
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
    
    
    
    parser.add_argument()
    
    return parser.parse_args()

def main(args):

    # dataset = load_dataset(args.dataset, args.subset).select_columns(
    #     [args.source_col, args.label_col]
    #     ).rename_columns(
    #         {args.source_col: 'function', args.label_col: 'label'}
    #         )
    
    dataset = load_from_disk(args.dataset_path).rename_columns(
        {
            args.source_col: 'function',
            args.label_col: 'label'
        }
    )
    
    
    dir_name = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    args.output_dir = os.path.join(args.output_dir, dir_name)

    
    if args.do_train:
        features = build_features(dataset, args)
        classifier = simple_classifier(max_iter=args.max_iter, random_state=args.random_state)
        results = {}
        
        for type in ['ec', 'eg', 'nf']:
            results[type] = classifier.train(features['train'][type], **{"output_dir": args.output_dir, 'dataset': args.dataset})
    
    if args.do_test:
        weight_paths = {
            'ec': os.path.join(args.weights_path, 'ec_weights.pkl'),
            'eg': os.path.join(args.weights_path, 'eg_weights.pkl'),
            'nf': os.path.join(args.weights_path, 'nf_weights.pkl')
        }
        features = {
            'ec': torch.load(os.path.join(args.features_path, 'test/ec.pt')),
            'eg': torch.load(os.path.join(args.features_path, 'test/eg.pt')),
            'nf': torch.load(os.path.join(args.features_path, 'test/nf.pt'))
        }
        
        y_test = np.array(dataset['test']['label'])
        
        for type, weight in weight_paths.items():
            print(simple_classifier.evaluate(
                weight,
                ds_sample(features[type], y_test, type)
            ))
            
        
        
        
        
        
    
if __name__ == '__main__':
    args = parse_args()
    main(args)