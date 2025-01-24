from . import OpenRouter, Openai, Claude, Google
from Generation.prompts import *

def sync_image_based(query, pages, model_type, prompt_template=IMAGE_PROMPT):
    provider = model_type.split("-")[0]
    model = model_type[len(provider) + 1:]

    if provider == "openai":
        return Openai.image_based_sync(query, pages, model, prompt_template)

def sync_text_based(query, chunks, model_type, prompt_template=TEXT_PROMPT):
    provider = model_type.split("-")[0]
    model = model_type[len(provider) + 1:]

    if provider == "openai":
        return Openai.text_based_sync(query, chunks, model, prompt_template)

def sync_hybrid(query, pages, chunks, model_type, prompt_template=HYBRID_PROMPT):
    provider = model_type.split("-")[0]
    model = model_type[len(provider) + 1:]

    if provider == "openai":
        return Openai.hybrid_sync(query, pages, chunks, model, prompt_template)

async def image_based(query, pages, model_type, prompt_template=IMAGE_PROMPT):
    provider = model_type.split("-")[0]
    model = model_type[len(provider)+1:]

    if model == "o1" or model == "gemini-2.0-flash-thinking-exp":
        prompt_template = O1_IMAGE_PROMPT

    if provider == "openai":
        return await Openai.image_based(query, pages, model, prompt_template)
    elif provider == "claude":
        return await Claude.image_based(query, pages, prompt_template)
    elif provider == "google":
        return await Google.image_based(query, pages, model, prompt_template)
    elif provider == "openrouter":
        return await OpenRouter.image_based(query, pages, model, prompt_template)

async def text_based(query, chunks, model_type, prompt_template=TEXT_PROMPT):
    provider = model_type.split("-")[0]
    model = model_type[len(provider)+1:]
    
    if model == "o1" or model == "gemini-2.0-flash-thinking-exp" or model == "deepseek/deepseek-chat":
        prompt_template = O1_TEXT_PROMPT

    if provider == "openai":
        return await Openai.text_based(query, chunks, model, prompt_template)
    elif provider == "claude":
        return await Claude.text_based(query, chunks, prompt_template)
    elif provider == "google":
        return await Google.text_based(query, chunks, model, prompt_template)
    elif provider == "openrouter":
        return await OpenRouter.text_based(query, chunks, model, prompt_template)

async def hybrid(query, pages, chunks, model_type, prompt_template=HYBRID_PROMPT):
    provider = model_type.split("-")[0]
    model = model_type[len(provider)+1:]

    if model == "o1" or model == "gemini-2.0-flash-thinking-exp":
        prompt_template = O1_HYBRID_PROMPT

    if provider == "openai":
        return await Openai.hybrid(query, pages, chunks, model, prompt_template)
    elif provider == "claude":
        return await Claude.hybrid(query, pages, chunks, prompt_template)
    elif provider == "google":
        return await Google.hybrid(query, pages, chunks, model, prompt_template)
    elif provider == "openrouter":
        return await OpenRouter.hybrid(query, pages, chunks, model, prompt_template)
