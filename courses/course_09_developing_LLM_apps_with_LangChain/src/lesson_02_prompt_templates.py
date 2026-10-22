"""
Prompt templates
    - Recipes for defining prompts for LLMs
    - Can contain: instructions, examples, and additional context
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_openai import ChatOpenAI
from common.config import OPENAI_API_KEY


template = "Explain this concept simply and concisely: {concept}"
prompt_template = PromptTemplate.from_template(
    template=template
)

prompt = prompt_template.invoke({"concept": "Prompting LLMs"})

print(prompt)

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation"
)
llm_chain = prompt_template | llm

concept = "Prompting LLMs"
print(llm_chain.invoke({"concept": concept}))

"""
- LangChain Expression Language (LCEL): | (pipe) operator
- Chain: connect calls to different components
"""

"""
Chat models
    - Chat roles: system, human, ai
"""

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a calculator that responds with math."),
        ("human", "Answer this math question: What is two plus two?"),
        ("ai", "2+2=4"),
        ("human", "Answer this math question: {math}")
    ]
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    max_tokens=50,
    temperature=2
)

llm_chain = template | llm
math="What is five times five?"

response = llm_chain.invoke({"math": math})
print(response.content)