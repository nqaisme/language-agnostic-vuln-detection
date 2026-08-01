from parser.input_extractor import gcb_input_extractor, cb_input_extractor
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, List, Any
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from dataclasses import dataclass
import torch, VARS, numpy as np, joblib, tqdm, os, datetime

class extractor:
    def __init__(self, model_name: str = 'microsoft/codebert-base', max_length: int = 512, batch_size: int = 64):
        
        try:
            assert all([
                model_name in ['microsoft/codebert-base', 'microsoft/graphcodebert-base'],
                max_length > 0 and batch_size > 0
            ])
        except:
            model_name = 'microsoft/codebert-base'
            max_length = 512
            batch_size = 64
            
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.model.to(self.device).eval()
    
    
    def __call__(self, source_codes: str| List[str], task: str = 'classificaiton') -> Tuple[torch.Tensor, torch.Tensor] | List[torch.Tensor]:

        try:
            assert task in ['classification', 'token_embedding']
        except:
            task = 'classification'
        if not source_codes:
            return torch.empty(0, self.max_length), torch.empty(0, self.max_length)
        
        match self.model_name:
            case VARS.GCB:
                ext = gcb_input_extractor(
                    tokenizer=self.tokenizer,
                    lang='c',
                    code_length=self.max_length,
                )
            case VARS.CB:
                ext = cb_input_extractor(
                    tokenizer=self.tokenizer,
                    code_length=self.max_length
                )
        
        if isinstance(source_codes, str): source_codes = [source_codes]

        results = []
        
        for i in tqdm.trange(0, len(source_codes), self.batch_size):
            
            batch_codes = source_codes[i: i + self.batch_size]
            
            inputs = ext(source_codes=batch_codes)
            inputs = {k : v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
        
            if task == 'classification':
                results.append(outputs.last_hidden_state[:, 0, :].cpu())
            
            else:
                batch_states = outputs.last_hidden_state.cpu()
                batch_masks = inputs['attention_mask'].cpu()
                
                for j in range(len(batch_codes)):
                    real_length = (batch_masks[j] != 0).sum().item()
                    results.append(batch_states[j, 1 : real_length - 1, :])

        if task == 'classification':
            return torch.cat(results, dim=0)
        return results


class neutral_feature_builder:
    def __init__(self, norm="l2", fuse="concat", alpha=0.5, pca_dim=None):
        self.norm = norm.lower() if norm else None
        self.fuse = fuse.lower()
        self.alpha = alpha
        self.pca_dim = pca_dim
        
        self.pca = PCA(n_components=pca_dim) if pca_dim is not None else None
        self.is_pca_fitted = False

    def _normalize(self, E: torch.Tensor) -> torch.Tensor:

        if self.norm is None:
            return E
        
        if self.norm == "l2":
            return F.normalize(E, p=2, dim=-1)
        
        elif self.norm == "minmax":
            min_val = E.min(dim=-1, keepdim=True).values
            max_val = E.max(dim=-1, keepdim=True).values
            eps = 1e-8
            return (E - min_val) / (max_val - min_val + eps)
        
        else:
            raise ValueError(f"The normalization method - {self.norm} is not supported!!\n")

    def fit_pca(self, F_raw_samples: torch.Tensor):
        if self.pca is not None:
            if isinstance(F_raw_samples, torch.Tensor):
                F_raw_samples = F_raw_samples.detach().cpu().numpy()
            
            self.pca.fit(F_raw_samples)
            self.is_pca_fitted = True

    def build(self, E_C: torch.Tensor, E_G: torch.Tensor) -> torch.Tensor:

        if E_C.shape[0] != E_G.shape[0] and E_C.dim() > 1:
            raise ValueError(f"Batch size mismatch: E_C has {E_C.shape[0]} samples, but E_G has {E_G.shape[0]} samples!!")

        is_1d = (E_C.dim() == 1)
        if is_1d:
            E_C = E_C.unsqueeze(0)
            E_G = E_G.unsqueeze(0)

        E_C_norm = self._normalize(E_C)
        E_G_norm = self._normalize(E_G)

        if self.fuse == "concat":
            F_raw = torch.cat([E_C_norm, E_G_norm], dim=-1)
            
        elif self.fuse == "average":
            if E_C_norm.shape != E_G_norm.shape:
                raise ValueError("The dimensions must match when performing AVERAGE fusion!!\n")
            F_raw = (E_C_norm + E_G_norm) / 2.0
            
        elif self.fuse == "weighted":
            if E_C_norm.shape != E_G_norm.shape:
                raise ValueError("The dimensions must match when performing WEIGHTED fusion!!\n")
            F_raw = self.alpha * E_C_norm + (1.0 - self.alpha) * E_G_norm
            
        else:
            raise ValueError(f"The fusion strategy - {self.fuse} is not supported!!\n")

        if self.pca_dim is not None:
            if not self.is_pca_fitted:
                raise RuntimeError("PCA has not been fitted yet! Please call fit_pca() before building!!\n")
            
            orig_device = F_raw.device
            orig_dtype = F_raw.dtype
            
            F_raw_np = F_raw.detach().cpu().numpy()
            F_np = self.pca.transform(F_raw_np)
            
            F = torch.from_numpy(F_np).to(device=orig_device, dtype=orig_dtype)
        else:
            F = F_raw

        if is_1d:
            F = F.squeeze(0)

        return F


class simple_classifier:
    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        self.random_state = random_state
        self.max_iter = max_iter
    
    def train(self, sample: ds_sample, **kwargs):
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight='balanced'
        )
        
        model.fit(sample.X, sample.y)
        
        
        output_dir = kwargs.get('output_dir', os.path.join(os.path.abspath, 'results'))
        file_name = f'{kwargs.get('dataset')}_{sample.type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_weights.pkl'
        joblib.dump(model, os.path.join(output_dir, file_name))
        
        print(f'model saving completed at {os.path.join(output_dir, file_name)}!\n')
        
        return os.path.join(output_dir, file_name)
    
    @staticmethod
    def evaluate(weights_file, X, y_true, average = 'binary'):
        model = joblib.load(weights_file)
        y_pred = model.predict(X)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
        

@dataclass
class ds_sample:
    X: Any
    y: Any
    type: str