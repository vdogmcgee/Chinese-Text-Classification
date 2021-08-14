![](https://img.shields.io/badge/license-MIT-blue.svg) 
![](https://img.shields.io/badge/Python-3.6.12-blue.svg)
![](https://img.shields.io/badge/torch-1.7.0-brightgreen.svg)
![](https://img.shields.io/badge/transformers-4.4.1-brightgreen.svg)
![](https://img.shields.io/badge/scikitlearn-0.24.0-brightgreen.svg)
![](https://img.shields.io/badge/tqdm-4.49.0-brightgreen.svg)
![](https://img.shields.io/badge/jsonlines-2.0.0-brightgreen.svg)
![](https://img.shields.io/badge/loguru-0.5.3-brightgreen.svg)



# Chinese-Text-Classification

新闻标题文本分类

- ### 1. 背景

  基于transformers bert做了一个文本分类的训练 + 推理流程

  ### 2. 文件

  ```shell
  > datasets		数据集文件夹
  > pretrained_model	各种预训练模型文件夹
  > saved_model		微调之后保存的模型文件夹
    train.py		训练文件
    inference.py		推理文件
  ```

  ### 3. 使用

  需要将公开数据集和预训练模型放到指定目录下， 并检查在代码中的位置是否对应

  ```python
  # 模型保存位置
  SAVE_PATH = 'saved_model/pytorch_model.bin'
  # 预训练模型目录
  BERT = 'pretrained_model/chinese_bert_pytorch'
  model_path = BERT
  # 数据位置
  CLASS_PATH = 'datasets/class.txt'
  TRAIN_PATH = 'datasets/train.txt'
  DEV_PATH = 'datasets/dev.txt'
  TEST_PATH = 'datasets/test.txt'
  ```

  训练

  ```shell
  python train.py
  ```

  推理

  ```shell
  python inference.py
  ```

  ### 4. 下载

  数据集：

  - 存放于项目datasets文件夹

  预训练模型：

  - BERT：https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
  - BERT-wwm：https://drive.google.com/file/d/1AQitrjbvCWc51SYiLN-cJq4e0WiNN4KY/view
  - BERT-wwm-ext：https://drive.google.com/file/d/1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_/view
  - RoBERTa-wwm-ext：https://drive.google.com/file/d/1eHM3l4fMo6DsQYGmey7UZGiTmQquHw25/view

  ### 5. 测评

  测评指标为acc

  参数：batch_size=128，lr=5e-5，maxlen=32，epochs=1（增加epochs可能效果会更好）

  | 模型            | dev    | test   |
  | :-------------- | ------ | ------ |
  | BERT            | 0.9351 | 0.9427 |
  | BERT-wwm        | 0.9367 | 0.9395 |
  | BERT-wwm-ext    | 0.9368 | 0.9419 |
  | RoBERTa-wwm-ext | 0.9385 | 0.9450 |

  ### 6. 参考

  - https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch







