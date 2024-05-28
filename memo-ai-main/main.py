import os
import openai
from dotenv import load_dotenv
load_dotenv()

from crewai import Crew, Agent, Task
from langchain_openai import ChatOpenAI
from textwrap import dedent

class Tasks(): 
    def taking_doubts(self, agent, biograpy, doubts):
        return Task(
                description=dedent(f"""\Você devera responder de forma humana a {doubts} referentes a o renomado cientista Stephen Halking, utilize as informações contidas no {biograpy}"""),
                expected_output=dedent("""\Responda todas as perguntas em portugues e somente perguntas relacionadas a infomrações sobre o Stephen Hawking"""),
                agent=agent
        )

class Agents():
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
    def agent_hawking(self): # Programando o agente.
        return Agent(
            name="AgentHawking",
            role="Educador",
            goal="Ensinar cosmologia e responder perguntas sobre a vida de Stephen Hawking.",
            backstory="Inspirado no renomado físico teórico Stephen Hawking, este agente é programado para compartilhar conhecimento detalhado sobre cosmologia e fatos da vida de Hawking. Utiliza um modelo de linguagem avançado para fornecer explicações claras e precisas em português.",
            verbose=False,
            llm=self.OpenAIGPT35,
            allow_delegation=False,
            language="pt"
        )
    
    
def extract_data_biograpy():
    file_path = 'biograpy.txt'

    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def halwking(doubts):  
    biograpy = extract_data_biograpy()
    
    agent = Agents()
    agent_hawking = agent.agent_hawking()
    
    tasks = Tasks()
    ask_questions = tasks.taking_doubts(agent_hawking, biograpy, doubts)
    
    crew = Crew(
        agents=[agent_hawking],
        tasks=[ask_questions]
    )
    result = crew.kickoff()
    return result
 

def main(): # Conversação.
    print("## Memória Biográfica de Stephen Hawking")
    print('---------------------------------------------')
    print("Agente Hawking: Olá meu caro, como posso ajudá-lo? Me pergunte qualquer coisa sobre o Stephen Halking.")
    
    while True:
        user_input = input("(Digite 'sair' para encerrar)User: ") # Solicitando a entrada do usuário
        if user_input.lower() == 'sair':
            print("Agente Halking: Até Logo!")
            break
        
        response = halwking(user_input) # Gerando a resposta do modelo
        print("Agente Halwking:", response) # Exibindo a resposta
        
if __name__ == "__main__":
    main()
