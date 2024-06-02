from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import (ChatPromptTemplate,FewShotChatMessagePromptTemplate)
from dotenv import load_dotenv
load_dotenv()
class ChatGenerator:
    def __init__(self) -> None:
        self.chat = ChatOpenAI(temperature=0,model_name = 'gpt-4-1106-preview')
        self.example =  [
            {
            "sentence" : "쪽바리앞잡이야 너거나라로 꺼져.",
            "masking_sentence" : "[mask]앞잡이야 [mask]나라로 [mask]",
            "answer":
            '''
            일본앞잡이야 너희들나라로 가라
            '''
            },
            {
            "sentence" : "미친 씨발 개슬람년아 그럼 한국에 오질 말든가",
            "masking_sentence" : "미친 [mask] [mask]슬람년아 그럼 한국에 오질 말든가",
            "answer":
            '''
            이슬람 여성들아 그럼 한국에 오질 말든가.
            '''
            },           
            {"sentence" : "음주운전하는 새끼들은 진짜 대가리에 뭐가 든건지... 다 무기징역 시켜라",
            "masking_sentence" : "음주운전하는 [mask] 진짜 [mask] 뭐가 든건지... 다 무기징역 시켜라",
            "answer":
            '''
            음주운전하는 사람들은 진짜 머리에 뭐가 든건지... 다 무기징역 시켜라
            '''
            },
            {
            "sentence" : "좌빨 영화 납시요 개돼지들 선동시키기 딱이요",
            "masking_sentence" : "[mask] 영화 납시요 [mask] [mask]시키기 딱이요",
            "answer":
            '''
            좌파 영화 납시요. 사람들 부추기기 딱이요
            '''
            }
        ]
        self.example_prompt = ChatPromptTemplate.from_messages(
                        [
                ("human", "원문 :{sentence}\n 마스킹 문장 : {masking_sentence}"),
                ("ai", "순화 문장 : {answer}"),
            ]
        )
        
        self.few_shot_prompt =  FewShotChatMessagePromptTemplate(
            example_prompt=self.example_prompt,
            examples=self.example,
        )
        self.final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "너는 문장을 순화하는 역할을 할거야. 원문과 함께 해당 문장이 유해한 문장으로 판단되는데 중요한 영향을 미친(=feature importance가 높은) 단어를 [mask] 처리한 마스킹 문장이 주어지면, 두 문장을 바탕으로 원문의 의미는 유지하면서 순화 댓글을 생성하는 것이 너의 임무야."),
                self.few_shot_prompt,
                ("human", "원문 :{sentence}\n 마스킹 문장 : {masking_sentence}"),
            ]
        )
        
        self.chain = self.final_prompt | self.chat
    def covert_sentence(self,ori_setence,mask_sentence):
        return self.chain.invoke({"sentence" : {ori_setence},
                                    "masking_sentence" : {mask_sentence}})