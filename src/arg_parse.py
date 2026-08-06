import argparse


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
        default='tensors',
    )
    
    parser.add_argument(
        '--ec',
        action='store_true',
        required=False
    )
    
    parser.add_argument(
        '--eg',
        action='store_true',
        required=False
    )
    
    parser.add_argument(
        '--all-embedding',
        action='store_true',
    )
    
    
    parser.add_argument(
        '--extract-only',
        action='store_true'
    )
    
    return parser.parse_args()