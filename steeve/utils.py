import logging, numpy as np
import torch
from typing import Tuple, List, Union, Optional
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.decomposition import PCA


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



PRETRAINED_MODELS = {
    'codebert': 'microsoft/codebert-base',
    'graphcodebert': 'microsoft/graphcodebert-base'
}

class DataPreProcess:
    def __init__(self, tokenizer: str = 'codebert', max_length: int = 512):

        assert tokenizer in PRETRAINED_MODELS.keys(), 'Value of `tokenizer` must be `codebert` or `graphcodebert`!'
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODELS.get(tokenizer))
        self.max_length = max_length


    def __call__(self, text: str | List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not text:
            logger.warning('\nInput text must not be empty!\n')
            return torch.empty(0, self.max_length), torch.empty(0, self.max_length)

        try:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )


            return tokens['input_ids'], tokens['attention_mask']

        except Exception as e:
            logger.error(f'\nFailed to tokenize input!\nError logs:\n{e}\n')
            raise


class EmbeddingExtractor:
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODELS.get('codebert'))

        self.codebert_model = AutoModel.from_pretrained(PRETRAINED_MODELS.get('codebert'))
        self.graph_codebert_model = AutoModel.from_pretrained(PRETRAINED_MODELS.get('graphcodebert'))

        logger.info('\nCodeBERT and GraphCodeBERT loaded successfully!\n')

        self.codebert_model.to(device)
        self.graph_codebert_model.to(device)

        self.codebert_model.eval()
        self.graph_codebert_model.eval()

        self.models = {
            'codebert': self.codebert_model,
            'graphcodebert': self.graph_codebert_model
        }

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, model_type: str = 'codebert', task: str = 'class') -> torch.Tensor:
        assert model_type in PRETRAINED_MODELS.keys(), 'Value of `model_type` must be in [`codebert`, `graphcodebert`]!'
        assert task in ('class', 'embedding'), 'Value of `task` must be in [`class`, `embedding`]'


        model = self.models.get(model_type)

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)

        with torch.no_grad():
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )

        return outputs.last_hidden_state.cpu() if task == 'class' else outputs.last_hidden_state[:, 0, :].cpu()


class NeutralFeatureBuilder:
    def __init__(self, norm: str = 'l2', fuse: str = 'concat', alpha: float = 0.5, pca_dim: int = None):

        if fuse not in ('concat', 'average', 'weighted'):
            raise ValueError("`fuse` must be in ['concat', 'average', 'weighted']!")
        if norm not in ('l2', 'minmax', None):
            raise ValueError("`norm` must be in ['l2', 'minmax', None]")

        self.norm = norm
        self.fuse = fuse
        self.alpha = alpha
        self.pca_dim = pca_dim

        self.pca = None # init after fitting PCA
        self.scaler = MinMaxScaler() if norm == 'minmax' else None


    def __call__(self,
                 E_C: np.ndarray,
                 E_G: np.ndarray,
                 fit_pca: bool = False) -> np.ndarray:
        E_C_norm = self._normalize(E_C)
        E_G_norm = self._normalize(E_G)

        F_raw = self._fuse(E_C_norm, E_G_norm)

        if self.pca_dim:
            if fit_pca: self.fit_pca(F_raw)
            return self.apply_pca(F_raw)
        else:
            return F_raw

    def _normalize(self, X: np.ndarray) -> np.ndarray:

        if self.norm == 'l2':
            return normalize(X, norm='l2', axis=1)

        if self.norm == 'minmax':
            if self.scaler is None:
                raise RuntimeError('MinMaxScaler has not been initialized!')
            return self.scaler.fit_transform(X)
        else:
            return X


    def _fuse(self, E_C: np.ndarray, E_G: np.ndarray) -> np.ndarray:
        if self.fuse == 'concat':
            return np.concatenate([E_C, E_G], axis=1)
        elif self.fuse == 'average':
            return (E_C + E_G) / 2
        elif self.fuse == 'weighted':
            return self.alpha * E_C + (1 - self.alpha) * E_G
        else:
            raise ValueError(f"Fusion strategy '{self.fuse}' is not supported!")

    def fit_pca(self, X: np.ndarray) -> None:
        if not self.pca_dim:
            logger.info('PCA has not been configed, fitting is skipped!')
            return

        self.pca = PCA(n_components=self.pca_dim, random_state=42)
        self.pca.fit(X)


    def apply_pca(self, X: np.ndarray) -> None:
        if self.pca is None:
            logger.warning('PCA has not been fitted, original data is returned!')
            return X
        return self.pca.transform(X)