from langchain_core.messages import HumanMessage
from .prompts import QAOutput, exponential_backoff
from langchain_anthropic import ChatAnthropic

with open(f"/Users/emrekuru/Developer/VRAG/.keys/claude_api_key.txt",  "r") as file:
    claude = file.read().strip()

async def image_based(query, pages, prompt_template):
    llm = ChatAnthropic(api_key=claude, model="claude-3-5-sonnet-20241022") 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = prompt_template.format(query=query, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]
    for p in pages:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg", 
                "data": p[0]
            }
        })

    message = HumanMessage(content=content)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response
    
async def text_based(query, chunks, prompt_template):
    llm =  ChatAnthropic(api_key=claude, model="claude-3-5-sonnet-20241022") 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = prompt_template.format(query=query, context=chunks, schema=schema)
    response: QAOutput = await exponential_backoff(llm.ainvoke, prompt)
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response

async def hybrid(query, pages, chunks, prompt_template):
    llm =  ChatAnthropic(api_key=claude, model="claude-3-5-sonnet-20241022") 
    llm = llm.with_structured_output(QAOutput)
    schema = QAOutput.model_json_schema()

    prompt = prompt_template.format(query=query, context=chunks, schema=schema)

    content = [
        {"type": "text", "text": prompt},
    ]

    for p in pages:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg", 
                "data": p[0]
            }
        })

    message = HumanMessage(content=content)
    response: QAOutput = await exponential_backoff(llm.ainvoke, [message])
    response = {"reasoning": response.reasoning, "answer": response.answer}
    return response