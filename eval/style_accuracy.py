import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'
import warnings
warnings.filterwarnings('ignore')

import shap
import scipy as sp

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import tqdm

# GPU 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# 평가 데이터셋 불러오기
eval_df = pd.read_excel("/home/xogns5037/KcELECTRA/datasets/Final_Evaluation.xlsx")

# 모델 불러오기
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)
model.load_state_dict(torch.load('/home/xogns5037/KcELECTRA/KcELECTRA_5_pth/checkpoint-4500/pytorch_model.bin'))

##### Sytle Accuracy #####
# 0: curse, 1: non_curse
def sentence_predict(sent):
    # 평가모드로 변경
    model.eval()

    # 입력된 문장 토크나이징
    tokenized_sent = tokenizer(
        sent,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        max_length=128
    )
    
    # 모델이 위치한 GPU로 이동 
    tokenized_sent.to(device)

    # 예측
    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_sent["input_ids"],
            attention_mask=tokenized_sent["attention_mask"],
            token_type_ids=tokenized_sent["token_type_ids"]
            )

    # 결과 return
    logits = outputs[0]
    logits = logits.detach().cpu()
    result = logits.argmax(-1)
    if result == 0: # 악성댓글
        result = 0
    elif result == 1: # 정상댓글
        result = 1
    return result


accuracy = {}

for column in eval_df.columns:
    
    pred = 0
    total = len(eval_df[column])

    for sent in eval_df[column]:
        result = sentence_predict(sent)
        # print(f"sentence: {sent}, result: {result}")
        if result == 0:
            continue
        else:
            pred += 1
    accuracy[column] = pred/total

print('-'*100)
print(accuracy)
print('-'*100)