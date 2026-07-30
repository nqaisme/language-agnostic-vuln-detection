import torch
import numpy as np
import tree_sitter_language_pack as tslp
from .utils import remove_comments_and_docstrings, tree_to_token_index, index_to_code_token
from .DFG import DFG_c_cpp


class cb_input_extractor:
    def __init__(self, tokenizer, code_length: int = 512):
        try:
            assert code_length > 0
        except:
            code_length = 512
        
        self.tokenizer = tokenizer
        self.code_length = code_length
    
    def __call__(self, source_code):
        
        inputs = self.tokenizer(
            source_code,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensor='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }


# extractor for GraphCodeBERT's inputs
# concludes: input_ids, position_ids, attn_mask
# ref: https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT

class gcb_input_extractor:
    def __init__(self, tokenizer, lang: str ='c', code_length: int = 512, data_flow_length: int = 64):
        
        try:
            assert all(
                lang in ['c', 'cpp'],
                code_length > 0 and data_flow_length > 0
            )
        except:
            lang = 'c'
            code_length = 512
            data_flow_length = 64

        self.tokenizer = tokenizer
        self.code_length = code_length
        self.data_flow_length = data_flow_length
        
        self.parser = tslp.get_parser(lang)
        self.lang = lang


    def extract_dataflow(self, code):
        try:
            code = remove_comments_and_docstrings(code, self.lang)
        except Exception:
            pass    
            
        try:
            tree = self.parser.parse(bytes(code, 'utf8'))    
            root_node = tree.root_node  
            tokens_index = tree_to_token_index(root_node)     
            code_lines = code.split('\n')
            code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]  
            
            index_to_code = {}
            for idx, (index, code_tok) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, code_tok)  
                
            try:
                DFG, _ = DFG_c_cpp(root_node, index_to_code, {}) 
            except Exception:
                DFG = []
                
            DFG = sorted(DFG, key=lambda x: x[1])
            indexs = set()
            for d in DFG:
                if len(d[-1]) != 0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG = []
            for d in DFG:
                if d[1] in indexs:
                    new_DFG.append(d)
            return code_tokens, new_DFG
        except Exception:
            return [], []

            
    def __call__(self, source_code):
        code_tokens, dfg = self.extract_dataflow(source_code)

        tokenized_code = [self.tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else self.tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
        
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(tokenized_code)):
            ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(tokenized_code[i]))
        
        flat_code_tokens = [y for x in tokenized_code for y in x]
        
        flat_code_tokens = flat_code_tokens[:self.code_length + self.data_flow_length - 3 - min(len(dfg), self.data_flow_length)][:512-3]
        source_tokens = [self.tokenizer.cls_token] + flat_code_tokens + [self.tokenizer.sep_token]
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        
        position_idx = [i + self.tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        
        dfg = dfg[:self.code_length + self.data_flow_length - len(source_tokens)]
        source_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg] 
        source_ids += [self.tokenizer.unk_token_id for x in dfg] 
        
        padding_length = self.code_length + self.data_flow_length - len(source_ids)
        position_idx += [self.tokenizer.pad_token_id] * padding_length
        source_ids += [self.tokenizer.pad_token_id] * padding_length
        
        
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
            
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([self.tokenizer.cls_token])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
        
        total_length = self.code_length + self.data_flow_length
        attn_mask = np.zeros((total_length, total_length), dtype=bool)
        
        node_index = sum([i > 1 for i in position_idx])
        max_length = sum([i != 1 for i in position_idx])
        
        attn_mask[:node_index, :node_index] = True
        
        for idx, i in enumerate(source_ids):
            if i in [0, 2]: 
                attn_mask[idx, :max_length] = True
                
        
        for idx, (a, b) in enumerate(dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
                
        
        for idx, nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(position_idx):
                    attn_mask[idx + node_index, a + node_index] = True
                    
        return {
            'input_ids': torch.tensor([source_ids]),
            'position_ids': torch.tensor([position_idx]),
            'attention_mask': torch.tensor([attn_mask])
        }