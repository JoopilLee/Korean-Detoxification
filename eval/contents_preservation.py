import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm


def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)  # 차원변환
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]  # l2 정규화 벡터크기 1로 만듬
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)).item()  # a,b 내적


model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", help="path to csv", required=True)
    args = parser.parse_args()
    mean_all = dict()
    for i in ['OurModel', 'wofewshot', 'woMask', 'Nothing']:
        mean_all[i] = []
    data = pd.read_csv(args.path)
    if len(data) > 50:
        for i in tqdm(range(0, len(data), 50), desc="Calculating scores"):
            batch_end = min(i + 50, len(data))
            batch_data = data[i:batch_end]
            names = ['raw', 'OurModel', 'wofewshot', 'woMask', 'Nothing']
            for name, i in zip(names, data.columns):
                globals()[name] = tokenizer(batch_data[i].tolist(), padding=True, truncation=True, return_tensors="pt")

            nv = [raw, OurModel, wofewshot, woMask, Nothing]
            vec = {}
            for i, key in tqdm(zip(nv, names), total=len(names), desc="Tokenizing data"):
                embeddings, _ = model(**i, return_dict=False)
                vec[key] = embeddings

            scores = []
            for i in tqdm(range(len(batch_data)), desc="Calculating scores"):
                score = []
                for n in names[1:]:
                    score.append(cal_score(vec['raw'][i][0], vec[n][i][0]))
                scores.append(score)

            scores = np.array(scores)
            print(names[1:])
            print(scores.mean(axis=0))
            for i, j in zip(names[1:], scores.mean(axis=0)):
                mean_all[i].append(j)

    for key, value_list in mean_all.items():
        avg_value = sum(value_list) / len(value_list) if len(
            value_list) > 0 else 0  # 값이 있는 경우에만 평균을 계산하고, 그렇지 않으면 0을 반환
        mean_all[key] = avg_value
    print(mean_all)
