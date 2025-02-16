from classifier import Classifier, SHAP
from chat_generate import ChatGenerator
# import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
generator = ChatGenerator()
model.load_state_dict(torch.load('./checkpoint-4500/pytorch_model.bin',map_location='cpu'), strict=False)
model.to(device)
cls = Classifier(tokenizer,model,device)
shap = SHAP(tokenizer,model,device)
print('-'*100)
# st.title("댓글 순화 :sunglasses:")
# 댓글 입력
# sentence = st.text_area("댓글을 입력해주세요: ")
# test_data = open('/home/jupyter/Korean_Text_Detoxification/results/samples.txt','r') 
# lines = test_data.readlines()
sentence = "각 집단마다 저런 쓰레기들 존나 많음.;"
print(tokenizer.tokenize(sentence))
# for line in lines:
#     print(line)
# sentence = input("댓글을 입력해주세요: ")

need_convert = False
if sentence:
    cls_result = cls.sentence_predict(sentence)
    if cls_result and '악성댓글' in cls_result:
        need_convert = True
        print(cls_result)
        # st.write(cls_result)
    else:
        # st.write(f'{cls_result}입니다. 다른 댓글을 입력해보세요.')
        print(f'{cls_result}입니다. 다른 댓글을 입력해보세요.')
        print(cls_result)
        need_convert = False

if need_convert:
    ori_sentence, masking_setence = shap.masking(sentence)
    # st.write(f'ori: {ori_sentence}')
    # st.write(f'mask : {masking_setence}')
    print(f'ori: {ori_sentence}')
    print(f'mask : {masking_setence}')

    converted_sentence = generator.covert_sentence(ori_sentence,masking_setence)
    print(f'기존 댓글 : {sentence}')
    print(f'{converted_sentence.content}')
    
    # if st.button('댓글 순화 요청하기'):
    #     with st.spinner('댓글 순화 요청하기'):
    #         converted_sentence = generator.covert_sentence(ori_sentence,masking_setence)
    #         st.write(f'기존 댓글 : {sentence}')
    #         st.write(f'{converted_sentence.content}')