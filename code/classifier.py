import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'
import warnings
warnings.filterwarnings('ignore')
import streamlit  as st
import shap
import scipy as sp

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ============================================< Classification >============================================
class Classifier:
    def __init__(self,tokenzier,model,device):         
        # GPU 설정
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

        # 토크나이징
        self.tokenizer = tokenzier

        # 모델 생성
        self.model = model

        #device 지정
        self.device = device
    
    # 모델 예측
    # 0: curse, 1: non_curse
    def sentence_predict(self,sent):
        # 평가모드로 변경
        self.model.eval()

        # 입력된 문장 토크나이징
        tokenized_sent = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
        )
        
        # 모델이 위치한 GPU로 이동 
        tokenized_sent.to(self.device)

        # 예측
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized_sent["input_ids"],
                attention_mask=tokenized_sent["attention_mask"],
                token_type_ids=tokenized_sent["token_type_ids"]
                )

        # 결과 return
        logits = outputs[0]
        logits = logits.detach().cpu()
        result = logits.argmax(-1)
        if result == 0:
            result = " >> 악성댓글 👿"
        elif result == 1:
            result = " >> 정상댓글 😀"
        return result

# ============================================< XAI >============================================
# SHAP
# define a prediction function
class SHAP:
    def __init__(self,tokenizer,model,device) -> None:
          self.tokenizer = tokenizer
          self.model = model
          self.explainer = shap.Explainer(self.f,self.tokenizer)
          self.device = device
          with open('/home/jupyter/Korean_Text_Detoxification/data/stopwords.txt', 'r', encoding='utf-8') as file:
            # 파일 내용을 줄 단위로 읽어와 리스트에 저장
            lines = file.readlines()
            self.stopwords = [line.strip() for line in lines]

    def f(self,x):
        tv = torch.tensor([self.tokenizer.encode(v, pad_to_max_length=True, max_length=128, truncation=True) for v in x],device=self.device)
        outputs = self.model(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:,1]) # use one vs rest logit units
        return val
    # build an explainer using a token masker
    def get_shapevalue(self,sentence):
        return self.explainer([sentence])

    def sort_shape_value(self,sentence,shap_values):
        print('-'*100)
        # 토큰화 결과
        print('토큰화 결과:', self.tokenizer.tokenize(sentence))
        # 부정적인 영향을 크게 미치는 순서대로 토큰 정렬
        print('-'*100)
        shap_values_list = list(zip(shap_values[0].values, shap_values[0].data))
        shap_values_list.sort(key=lambda x: x[0])
        for shap_value, feature in shap_values_list:
            if shap_value == 0.0:
                continue
            else:
                print(feature, shap_value)
        print('-'*100)
         
    def masking(self,sentence):
        shap_values = self.explainer([sentence])
        threshold = sum([i for i in shap_values[0].values if i < 0]) / len([i for i in shap_values[0].values if i < 0])
        # 마스킹
        mask_list = []
        shap_values_zip = zip(shap_values[0].values, shap_values[0].data)
        for shap_value, feature in shap_values_zip:
            if shap_value < threshold: # threshold
                if feature.strip() not in self.stopwords: # stopwords 제거
                    if feature.strip().isalnum(): # 특수문자 제거
                        mask_list.append(feature.strip())
        print('-'*100)
        print('masking list:', mask_list)
        print(f"원본 문장 : {sentence}")
        ori = sentence
        for mask in mask_list:
            sentence = sentence.replace(mask, "[mask]")

        return ori, sentence


# ============================================< Masking >============================================
