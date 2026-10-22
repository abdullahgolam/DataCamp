# from langchain_openai import ChatOpenAI
# from common.config import OPENAI_API_KEY
# from common.helpers import print_llm_output
#
#
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     api_key=OPENAI_API_KEY,
#     max_tokens=50,
#     temperature=2
# )
#
# output = llm.invoke("what is langChain?")
#
# print_llm_output(output)
#
#
#
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from common.helpers import print_llm_output


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100, "return_full_text": False}
)
chat = ChatHuggingFace(llm=llm)

print_llm_output(chat.invoke("What is Hugging Face?"))
