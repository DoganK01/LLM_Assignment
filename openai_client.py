import os
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel
from openai import AzureOpenAI

from config import PipelineConfig


def recursive_as_dict(obj):
    if isinstance(obj, list):
        return [recursive_as_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: recursive_as_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return recursive_as_dict(obj.__dict__)
    else:
        return obj

class ChatCompletionMessage(BaseModel):
    content: str
    refusal: Optional[str] = None
    role: str
    audio: Optional[Any] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[Dict[str, Any]] = None

class Choice(BaseModel):
    finish_reason: Optional[str]
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    message: ChatCompletionMessage
    content_filter_results: Optional[Dict[str, Any]] = None

class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: int
    audio_tokens: int
    reasoning_tokens: int
    rejected_prediction_tokens: int

class PromptTokensDetails(BaseModel):
    audio_tokens: int
    cached_tokens: int

class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails
    prompt_tokens_details: PromptTokensDetails

class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: str
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None
    usage: CompletionUsage
    prompt_filter_results: Optional[List[Dict[str, Any]]] = None

class EmbeddingData(BaseModel):
    object: str
    index: int
    embedding: List[float]

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingsResponse(BaseModel):
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Usage


def init_openai_client(config: PipelineConfig) -> AzureOpenAI:
    """
    Initializes the AzureOpenAI client using environment variables.
    """
    if not config.openai_api_key or not config.azure_endpoint:
        raise ValueError("Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT in environment variables.")
    client_text = AzureOpenAI(
        api_key=config.openai_api_key,
        api_version=config.api_version,
        azure_endpoint=config.azure_endpoint
    )
    client_embedding = AzureOpenAI(
        api_key=config.openai_embedding_api_key,
        api_version=config.embedding_api_version,
        azure_endpoint=config.azure_embedding_endpoint
    )
    logging.info("AzureOpenAI client initialized.")
    return client_text, client_embedding


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = PipelineConfig()
    client_text, client_embedding = init_openai_client(config)
    deployment_name = "gpt-4o"

    logging.info("Sending a test completion job")
    start_phrase = "Write a tagline for an ice cream shop. "
    try:
        response = client_text.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": "Who were the founders of Microsoft?"}
            ]
        )
        response_dict = recursive_as_dict(response)
        chat_response = ChatCompletionResponse.model_validate(response_dict)
        print(chat_response.model_dump_json(indent=2))
        print(chat_response.choices[0].message.content)
    except Exception as e:
        logging.error("Error during chat completion", exc_info=e)

    try:
        embedding_response = client_embedding.embeddings.create(
            input="Your text string goes here",
            model="text-embedding-ada-002"
        )
        print(embedding_response)
        emb_response_dict = recursive_as_dict(embedding_response)
        emb_response = EmbeddingsResponse.model_validate(emb_response_dict)
        print(emb_response.model_dump_json(indent=2))
        print("Embedding vector:")
        for value in emb_response.data[0].embedding:
            print(value)
    except Exception as e:
        logging.error("Error during embedding creation", exc_info=e)
