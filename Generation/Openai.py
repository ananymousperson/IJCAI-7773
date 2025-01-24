from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from .prompts import QAOutput, exponential_backoff, exponential_backoff_sync
import os 
from langchain_community.callbacks import get_openai_callback

with open(f".keys/openai_api_key.txt", "r") as file:
    os.environ["OPENAI_API_KEY"] = file.read().strip()
    api_key = file.read().strip()

async def image_based(query, pages, model_type, prompt_template):
    llm = ChatOpenAI(model=model_type, api_key=api_key, disabled_params={"parallel_tool_calls": None}) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = prompt_template.format(query=query, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]

    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    message = HumanMessage(content=content)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response

def image_based_sync(query, pages, model_type, prompt_template):
    llm = ChatOpenAI(model=model_type, api_key=api_key, disabled_params={"parallel_tool_calls": None}) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = prompt_template.format(query=query, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]

    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    message = HumanMessage(content=content)
    response: QAOutput = exponential_backoff_sync(llm.invoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response
    
async def text_based(query, chunks, model_type, prompt_template):
    llm = ChatOpenAI(model=model_type, api_key=api_key, disabled_params={"parallel_tool_calls": None}) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()    

    prompt = prompt_template.format(query=query, context=chunks, schema=schema)

    response: QAOutput = await exponential_backoff(llm.ainvoke, prompt)
    response = {"reasoning": response.reasoning, "answer": response.answer}                                    
    return response

def text_based_sync(query, chunks, model_type, prompt_template):
    llm = ChatOpenAI(model=model_type, api_key=api_key, disabled_params={"parallel_tool_calls": None}) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()    

    prompt = prompt_template.format(query=query, context=chunks, schema=schema)

    response: QAOutput = exponential_backoff_sync(llm.invoke, prompt)
    response = {"reasoning": response.reasoning, "answer": response.answer}                                    
    return response

async def hybrid(query, pages, chunks, model_type, prompt_template):
    llm = ChatOpenAI(model=model_type, api_key=api_key, disabled_params={"parallel_tool_calls": None}) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()    

    prompt = prompt_template.format(query=query, context=chunks, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]
    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    message = HumanMessage(content=content)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response

def hybrid_sync(query, pages, chunks, model_type, prompt_template):
    llm = ChatOpenAI(model=model_type, api_key=api_key, disabled_params={"parallel_tool_calls": None}) 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()    

    prompt = prompt_template.format(query=query, context=chunks, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]
    for p in pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{p}"}
        })

    message = HumanMessage(content=content)
    response: QAOutput = exponential_backoff_sync(llm.invoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response
