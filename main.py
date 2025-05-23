from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2:latest")

template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)

# Monitor the Memory in Cuda via - Nvidia-smi
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0 = first GPU
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"Total GPU memory: {mem_info.total / 1024**2:.2f} MB")
print(f"Free GPU memory: {mem_info.free / 1024**2:.2f} MB")
print(f"Used GPU memory: {mem_info.used / 1024**2:.2f} MB")