from classifier import Classifier, SHAP
from chat_generate import ChatGenerator
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

# 댓글 입력
test_data = open('/home/jupyter/Korean_Text_Detoxification/data/toxic.txt','r') 
lines = test_data.readlines()
output_file_path = '/home/jupyter/Korean_Text_Detoxification/results/Final2.txt'
converted_sentences = []

for sentence in lines:
    need_convert = False
    if sentence:
        cls_result = cls.sentence_predict(sentence)
        if cls_result and '악성댓글' in cls_result:
            need_convert = True
            print(cls_result)
        else:
            print(f'{cls_result}입니다. 다른 댓글을 입력해보세요.')
            print(cls_result)
            need_convert = False

    if need_convert:
        ori_sentence, masking_setence = shap.masking(sentence)
        print(f'ori: {ori_sentence}')
        print(f'mask : {masking_setence}')

        converted_sentence = generator.covert_sentence(ori_sentence,masking_setence)
        print(f'기존 댓글 : {ori_sentence}')
        print(f'{converted_sentence.content}')
        
        # converted_sentences.extend((ori_sentence,'\n',converted_sentence.content,'\n','-'*100))
        converted_sentences.append(converted_sentence.content)
        
with open(output_file_path,'w') as output_file:
    for con_sentence in converted_sentences:
        output_file.write(con_sentence + '\n')