import numpy as np, argparse
import tree_sitter_language_pack as tslp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import load_from_disk
from parser.utils import remove_comments_and_docstrings

parser = tslp.get_parser('c')

def evaluate(y_pred, y_true, average='binary'):
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1': round(f1, 2)
    }

def traverse_ast(node, current_depth=1):
    total_nodes = 1
    max_depth = current_depth
    
    num_if = 1 if node.type == 'if_statement' else 0
    num_loops = 1 if node.type in ['for_statement', 'while_statement', 'do_statement'] else 0
    num_calls = 1 if node.type == 'call_expression' else 0

    for child in node.children:
        c_nodes, c_depth, c_if, c_loops, c_calls = traverse_ast(child, current_depth + 1)
        total_nodes += c_nodes
        max_depth = max(max_depth, c_depth)
        num_if += c_if
        num_loops += c_loops
        num_calls += c_calls

    return total_nodes, max_depth, num_if, num_loops, num_calls

def extract_ast_features(code_str):
    try:
        tree = parser.parse(bytes(code_str, "utf8"))
        root_node = tree.root_node
        return list(traverse_ast(root_node))
    except:
        return [0, 0, 0, 0, 0]
    
def parse_arg():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--source-col',
        type=str,
        required=True
    )
    parser.add_argument(
        '--label-col',
        type=str,
        required=True
    )
    
    return parser.parse_args()

def main(args):
    dataset = load_from_disk(args.dataset_dir).select_columns([args.source_col, args.label_col])
    
    ast_train = [extract_ast_features(remove_comments_and_docstrings(code, 'c')) for code in list(dataset['train'][args.source_col])]
    ast_test = [extract_ast_features(remove_comments_and_docstrings(code, 'c')) for code in list(dataset['test'][args.source_col])]
    
    ast_train, ast_test = np.array(ast_train), np.array(ast_test)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    tf_idf_train = vectorizer.fit_transform(list(dataset['train'][args.source_col]))
    tf_idf_test = vectorizer.transform(list(dataset['test'][args.source_col]))
    
    scaler = StandardScaler()
    ast_train_scaled = scaler.fit_transform(ast_train)
    ast_test_scaled = scaler.transform(ast_test)
    
    X_train = hstack([tf_idf_train, ast_train_scaled])
    X_test = hstack([tf_idf_test, ast_test_scaled])
    
    model = LinearSVC()
    model.fit(X_train, list(dataset['train'][args.label_col]))
    predictions = model.predict(X_test)
    
    y_pred = model.predict(X_test)
    results = evaluate(y_pred, np.array(dataset['test'][args.label_col]))
    print(f"Precision: {results['precision']}, Recall: {results['recall']}, F1 Score: {results['f1']}")
    
if __name__ == "__main__":
    args = parse_arg()
    main(args)