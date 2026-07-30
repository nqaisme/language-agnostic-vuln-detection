from parser.input_extractor import gcb_input_extractor, cb_input_extractor
from transformers import AutoTokenizer, AutoModel
import torch, logging, VARS
from typing import Tuple, List
logging

class extractor:

    def __init__(self, model_name: str = 'microsoft/codebert-base', max_length: int = 512, batch_size: int = 64):
        
        try:
            assert all(
                model_name in ['microsoft/codebert-base', 'microsoft/graphcodebert-base'],
                max_length > 0 and batch_size > 0
            )
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
        
        self.model.eval()
    
    
    def extract(self, source_codes: str| List[str], task: str = 'classificaiton') -> Tuple[torch.Tensor, torch.Tensor]:

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
        inputs = ext(source_code=source_codes)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )
        
        if task == 'classification':
            return outputs.last_hidden_states[:, 0, :]
        
        # ignoring [PAD token
        real_token_length = inputs['attention_mask'][0].sum().item()
        # drop <s> and </s>
        return outputs.last_hidden_states[:, 1: real_token_length - 1, :]


            
            
