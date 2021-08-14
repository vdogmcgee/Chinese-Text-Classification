# -*- encoding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, BertTokenizer

from train import TARGET_NAMES, TextDataset, TextClassifyModel, eval

BATCH_SIZE = 64
MAXLEN = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

SAVE_DIR = 'saved_model'
SAVE_CKPT = 'saved_model/pytorch_model.bin'


class InferDataset(Dataset):
    """自定义数据集"""
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text):
        return self.tokenizer(text, max_length=MAXLEN, truncation=True, 
                              padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        return self.text_2_id(self.data[index])
    
    
def infer(model, dataloader):
    """推理函数"""
    model.eval()    
    preds = torch.tensor([], dtype=int, device=DEVICE)
    with torch.no_grad():
        for text in dataloader: 
            input_ids = text.get('input_ids').squeeze(1).to(DEVICE)
            attention_mask = text.get('attention_mask').squeeze(1).to(DEVICE)
            token_type_ids = text.get('token_type_ids').squeeze(1).to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids)  # [batch, 10]
            max_idx = torch.max(out, 1)[1]
            preds = torch.cat((preds, max_idx), dim=0)
    return [TARGET_NAMES[i] for i in preds.cpu().numpy()]
    

if __name__ == '__main__':
            
    texts = [
        '今年春晚节目单出来了',
        '刘翔夺冠了',
        '基金今天涨了5个点',
        '国家总统访问美国',
        '中小学马上要开学了',
        '男子救人落水不幸逝世',
        '白酒股最近跌了很多',
        '吴亦凡入狱了, 大快人心!',
    ]
    # load data
    tokenizer = BertTokenizer.from_pretrained(SAVE_DIR)
    dataloader = DataLoader(InferDataset(texts, tokenizer), batch_size=BATCH_SIZE)
    # load model
    model = TextClassifyModel(pretrained_model=SAVE_DIR).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_CKPT))
    # infer
    res = infer(model, dataloader)
    for text, pred in zip(texts, res):
        print(f'{text} {pred}')
        