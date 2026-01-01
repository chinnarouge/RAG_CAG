import ollama
from functools import lru_cache

DEFAULT_MODEL = "gpt-oss"

PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local)",
        "models": ["gpt-oss", "llama3.2", "mistral", "qwen3", "deepseek-r1:1.5b"],
        "requires_key": False
    },
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "requires_key": True
    },
    "azure": {
        "name": "Azure OpenAI",
        "models": ["gpt-4", "gpt-4o", "gpt-35-turbo"],
        "requires_key": True,
        "requires_endpoint": True
    },
    "gemini": {
        "name": "Google Gemini",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "requires_key": True
    },
    "claude": {
        "name": "Anthropic Claude",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "requires_key": True
    }
}

_openai_clients = {}
_azure_clients = {}
_gemini_configured = {}
_gemini_models = {}
_anthropic_clients = {}


def get_providers() -> dict:
    return PROVIDERS


@lru_cache(maxsize=1)
def get_available_models(provider: str = "ollama") -> list[str]:
    if provider == "ollama":
        try:
            models = ollama.list()
            installed = [m["name"].split(":")[0] for m in models.get("models", [])]
            return tuple(installed) if installed else tuple(PROVIDERS["ollama"]["models"])
        except:
            return tuple(PROVIDERS["ollama"]["models"])
    return tuple(PROVIDERS.get(provider, {}).get("models", []))


def _get_openai_client(api_key: str):
    if api_key not in _openai_clients:
        from openai import OpenAI
        _openai_clients[api_key] = OpenAI(api_key=api_key)
    return _openai_clients[api_key]


def _generate_with_openai(prompt: str, model: str, api_key: str, stream: bool = False):
    client = _get_openai_client(api_key)
    
    if stream:
        def _stream():
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return _stream()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


def _get_azure_client(api_key: str, endpoint: str):
    cache_key = f"{api_key}_{endpoint}"
    if cache_key not in _azure_clients:
        from openai import AzureOpenAI
        _azure_clients[cache_key] = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint
        )
    return _azure_clients[cache_key]


def _generate_with_azure(prompt: str, model: str, api_key: str, endpoint: str, stream: bool = False):
    client = _get_azure_client(api_key, endpoint)
    
    if stream:
        def _stream():
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return _stream()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


def _configure_gemini(api_key: str):
    if api_key not in _gemini_configured:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_configured[api_key] = True
    return True


def _get_gemini_model(api_key: str, model: str):
    _configure_gemini(api_key)
    cache_key = f"{api_key}:{model}"
    if cache_key not in _gemini_models:
        import google.generativeai as genai
        _gemini_models[cache_key] = genai.GenerativeModel(model)
    return _gemini_models[cache_key]


def _generate_with_gemini(prompt: str, model: str, api_key: str, stream: bool = False):
    gen_model = _get_gemini_model(api_key, model)
    
    if stream:
        def _stream():
            response = gen_model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        return _stream()
    else:
        response = gen_model.generate_content(prompt)
        return response.text


def _get_anthropic_client(api_key: str):
    if api_key not in _anthropic_clients:
        import anthropic
        _anthropic_clients[api_key] = anthropic.Anthropic(api_key=api_key)
    return _anthropic_clients[api_key]


def _generate_with_claude(prompt: str, model: str, api_key: str, stream: bool = False):
    client = _get_anthropic_client(api_key)
    
    if stream:
        def _stream():
            with client.messages.stream(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        return _stream()
    else:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


def _generate_with_ollama(prompt: str, model: str, stream: bool = False):
    if stream:
        def _stream():
            for chunk in ollama.generate(model=model, prompt=prompt, stream=True):
                yield chunk["response"]
        return _stream()
    else:
        response = ollama.generate(model=model, prompt=prompt)
        return response["response"]


def _build_prompt(query: str, context: str, conversation_history: list = None) -> str:
    if not context or context.strip() == "":
        return f"""You are a helpful assistant for a RAG system.

The user asked: {query}

No documents have been uploaded yet. Please inform the user to upload documents first using the sidebar."""

    conv_context = ""
    if conversation_history:
        conv_context = "\nPrevious conversation:\n"
        for msg in conversation_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv_context += f"{role}: {msg['content'][:500]}\n"

    return f"""You are a helpful assistant.
Answer the question based on the context below.
If the answer is not clearly in the context, say so but try to be helpful.

Document Context:
{context}
{conv_context}
Current Question:
{query}
"""


def generate_answer(
    query: str, 
    context: str, 
    model: str = None,
    provider: str = "ollama",
    api_key: str = None,
    azure_endpoint: str = None,
    conversation_history: list = None
) -> str:
    model = model or DEFAULT_MODEL
    prompt = _build_prompt(query, context, conversation_history)
    
    try:
        if provider == "openai" and api_key:
            return _generate_with_openai(prompt, model, api_key, stream=False)
        elif provider == "azure" and api_key and azure_endpoint:
            return _generate_with_azure(prompt, model, api_key, azure_endpoint, stream=False)
        elif provider == "gemini" and api_key:
            return _generate_with_gemini(prompt, model, api_key, stream=False)
        elif provider == "claude" and api_key:
            return _generate_with_claude(prompt, model, api_key, stream=False)
        else:
            return _generate_with_ollama(prompt, model, stream=False)
    except Exception as e:
        return f"Error generating response: {str(e)}"


def generate_answer_stream(
    query: str, 
    context: str, 
    model: str = None,
    provider: str = "ollama",
    api_key: str = None,
    azure_endpoint: str = None,
    conversation_history: list = None
):
    model = model or DEFAULT_MODEL
    prompt = _build_prompt(query, context, conversation_history)
    
    try:
        if provider == "openai" and api_key:
            yield from _generate_with_openai(prompt, model, api_key, stream=True)
        elif provider == "azure" and api_key and azure_endpoint:
            yield from _generate_with_azure(prompt, model, api_key, azure_endpoint, stream=True)
        elif provider == "gemini" and api_key:
            yield from _generate_with_gemini(prompt, model, api_key, stream=True)
        elif provider == "claude" and api_key:
            yield from _generate_with_claude(prompt, model, api_key, stream=True)
        else:
            yield from _generate_with_ollama(prompt, model, stream=True)
    except Exception as e:
        yield f"Error: {str(e)}"
