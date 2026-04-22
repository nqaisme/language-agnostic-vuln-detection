import logging
from typing import Tuple, List, Union, Optional
from transformers import AutoTokenizer, AutoModel
import torch


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
