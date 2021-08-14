# -*- encoding: utf-8 -*-

import random
import time
from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

# 基本参数
EPOCHS = 1
BATCH_SIZE = 128
LR = 5e-5
MAXLEN = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# 模型位置
SAVE_PATH = 'saved_model/pytorch_model.bin'
BERT = 'pretrained_model/chinese_bert_pytorch'
model_path = BERT

# 数据位置
CLASS_PATH = 'datasets/class.txt'
TARGET_NAMES = [x.strip() for x in open(CLASS_PATH).readlines()]
NUMBER_CLASS = len(TARGET_NAMES)
TRAIN_PATH = 'datasets/train.txt'
DEV_PATH = 'datasets/dev.txt'
TEST_PATH = 'datasets/test.txt'


def load_data(path):
    """加载数据"""
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            line = line.strip()
            text, label = line.split('\t')
            data.append((text, label))
    return data
    
    
class TextDataset(Dataset):
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
        return self.text_2_id(self.data[index][0]), int(self.data[index][1])
    
    
class TextClassifyModel(nn.Module):
    """文本分类模型"""
    def __init__(self, pretrained_model: str):
        super(TextClassifyModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(config.hidden_size, NUMBER_CLASS)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids)
        pooler = out.pooler_output   # [batch, 768]
        out = self.fc(pooler)
        return out
    
            
def compute_loss(pred, label):
    """自定义损失函数"""
    return F.cross_entropy(pred, label)
    
    
def eval(model, dataloader, test_flag=False) -> Union[float, Tuple]:
    """评估函数"""
    model.eval()
    preds = torch.tensor([], dtype=int, device=DEVICE)
    labels = np.array([], dtype=int)
    with torch.no_grad():
        for text, label in dataloader: 
            input_ids = text.get('input_ids').squeeze(1).to(DEVICE)
            attention_mask = text.get('attention_mask').squeeze(1).to(DEVICE)
            token_type_ids = text.get('token_type_ids').squeeze(1).to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids)
            pred = torch.max(out, 1)[1]
            preds = torch.cat((preds, pred), dim=-1)
            labels = np.append(labels, label) 
    preds = preds.cpu().numpy()
    acc = metrics.accuracy_score(labels, preds)   # 避免频繁的cpu与gpu的数据交换
    if not test_flag:
        return acc
    report = metrics.classification_report(labels, preds, target_names=TARGET_NAMES, digits=4)
    confusion = metrics.confusion_matrix(labels, preds)
    return acc, report, confusion

    
def train(model, train_dl, dev_dl, optimizer) -> None:
    """训练函数"""
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, (text, label) in enumerate(tqdm(train_dl), start=1):   
        # [batch, 1, seq_len] -> [batch, seq_len]
        input_ids = text.get('input_ids').squeeze(1).to(DEVICE)
        attention_mask = text.get('attention_mask').squeeze(1).to(DEVICE)
        token_type_ids = text.get('token_type_ids').squeeze(1).to(DEVICE)
        label = label.to(DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)  # [batch, num_class]                
        loss = compute_loss(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 100 == 0:
            logger.info(f'loss: {loss.item():.4f}')            
            _, indices = torch.max(out, dim=1)
            
            dev_acc = eval(model, dev_dl)
            model.train()
            if best < dev_acc:
                best = dev_acc
                early_stop_batch = 0
                torch.save(model.state_dict(), SAVE_PATH)
                logger.info(f"higher dev acc: {best:.4f} in batch: {batch_idx}, save model")
                continue
            early_stop_batch += 1
            if early_stop_batch == 30:
                logger.info(f"dev acc doesn't improve for {early_stop_batch} batch, early stop!")
                return 
            
    
if __name__ == '__main__':
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load data
    train_data = load_data(TRAIN_PATH)
    dev_data = load_data(DEV_PATH)
    test_data = load_data(TEST_PATH)
    train_dataloader = DataLoader(TextDataset(train_data, tokenizer), batch_size=BATCH_SIZE)
    dev_dataloader = DataLoader(TextDataset(dev_data, tokenizer), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(TextDataset(test_data, tokenizer), batch_size=BATCH_SIZE)
    # load model
    model = TextClassifyModel(pretrained_model=model_path)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train 
    best = 0
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')    
        train(model, train_dataloader, dev_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    # eval
    model.load_state_dict(torch.load(SAVE_PATH))
    test_acc, test_report, test_confusion = eval(model, test_dataloader, test_flag=True)
    print(f'acc: {test_acc:.4f}')
    print(f'classification report:')
    print(f'{test_report}') 
    print(f'confusion matrix:')
    print(f'{test_confusion}')
    
