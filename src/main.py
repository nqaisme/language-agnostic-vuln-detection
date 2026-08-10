from model import extractor, neutral_feature_builder, simple_classifier, ds_sample
import VARS, numpy as np, os, torch, datetime
from tmp import tmp_dataset
from arg_parse import parse_args
from datasets import load_from_disk

def save_tensor(tensor: dict, split: str, type: str, args):
    if not args.result_dir:
        result_dir = os.path.join(os.getcwd(), 'tensors')
    else:
        result_dir = os.path.join(args.result_dir, 'tensors')

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
    
    if args.extract_only:
        
        if args.ec:
            [save_tensor(cx(list(dataset[split]['function'])), 'ec', args) for split in ['train', 'test']]
        elif args.eg:
            [save_tensor(gcx(list(dataset[split]['function'])), 'eg', args) for split in ['train','test']]
        elif args.all_embedding:
            [save_tensor(cx(list(dataset[split]['function'])), 'ec', args) for split in ['train', 'test']]
            [save_tensor(gcx(list(dataset[split]['function'])), 'eg', args) for split in ['train','test']]
    
    else:
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
    
    

    
    if args.do_train:
        subdir_name = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        args.result_dir = os.path.join(args.result_dir, subdir_name)
        
        with open(os.path.join(args.result_dir, 'note.txt'), 'w') as f:
            f.write(f"dataset: {args.dataset_dir}\ntime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        features = build_features(dataset.shuffle(seed=42), args)
        classifier = simple_classifier(max_iter=args.max_iter, random_state=args.random_state)
        results = {}
        
        for type in ['ec', 'eg', 'nf']:
            results[type] = classifier.train(features['train'][type], **{"output_dir": args.result_dir, 'dataset': os.path.basename(args.dataset_dir)})
    
    elif args.do_test:
        weight_paths = {
            'ec': os.path.join(args.result_dir, args.result_subdir, 'ec_weights.pkl'),
            'eg': os.path.join(args.result_dir, args.result_subdir, 'eg_weights.pkl'),
            'nf': os.path.join(args.result_dir, args.result_subdir, 'nf_weights.pkl'),            
        }
        tensor_paths = {
            'ec': os.path.join(args.result_dir, args.result_subdir, args.tensor_dir, 'test/ec.pt'),
            'eg': os.path.join(args.result_dir, args.result_subdir, args.tensor_dir, 'test/eg.pt'),
            'nf': os.path.join(args.result_dir, args.result_subdir, args.tensor_dir, 'test/nf.pt'),
        }
        
        y_true = np.array(dataset['test']['label'])
        
        for type, weight_file in weight_paths.items():
            sample = ds_sample(
                torch.load(tensor_paths[type]),
                y_true,
                type
            )
            print(f"{type} embedding result:\t{simple_classifier.evaluate(weight_file, sample)}")
    
    elif args.extract_only:
        build_features(dataset, args)
if __name__ == '__main__':
    args = parse_args()
    main(args)