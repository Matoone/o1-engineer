# model_manager.py
from typing import Dict, List, Optional, Tuple, Any, Union, cast
import os
import logging
from enum import Enum
import json
from anthropic import AsyncAnthropic
import ollama
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class ModelError(Exception):
    """Base exception class for model-related errors"""

    pass


class ModelConfigurationError(ModelError):
    """Raised when there's an error in model configuration"""

    pass


class ModelAPIError(ModelError):
    """Raised when there's an error in API calls"""

    pass


class ModelNotFoundError(ModelError):
    """Raised when specified model is not found in configuration"""

    pass


# Configuration des modèles
MODEL_CONFIG = {
    # Anthropic Claude3.5 Sonnet
    "anthropic/claude-3-5-sonnet-latest": {
        "max_tokens": 8192,
        "supports_tools": True,
        "streaming": True,
        "requires_api_key": True,
        "env_key": "ANTHROPIC_API_KEY",
    },
    # Ollama Qwen coder 14b
    "ollama/qwen2.5-coder:14b": {
        "max_tokens": 8192,
        "supports_tools": True,
        "streaming": True,
        "requires_api_key": False,
        "env_key": None,
    },
    # OpenAI GPT4o
    "openai/gpt-4o": {
        "max_tokens": 8192,
        "supports_tools": True,
        "streaming": True,
        "requires_api_key": True,
        "env_key": "OPENAI_API_KEY",
    },
}


def parse_model_name(self, model_name: str) -> Tuple[ModelProvider, str]:
    """
    Extrait le provider et le nom du modèle
    """
    try:
        provider, model = model_name.split("/", 1)
        return ModelProvider(provider), model
    except ValueError:
        raise ModelConfigurationError(f"Invalid model name format: {model_name}")


def load_model_config() -> str:
    """
    Charge la configuration du modèle depuis les variables d'environnement
    """
    load_dotenv()
    model = os.getenv("MODEL")
    if not model:
        raise ModelConfigurationError("MODEL not found in environment variables")
    if model not in MODEL_CONFIG:
        raise ModelNotFoundError(f"Model {model} not found in configuration")
    return model


def validate_api_keys(model_name: str) -> None:
    """
    Vérifie que les clés API nécessaires sont présentes
    """
    config = MODEL_CONFIG.get(model_name)
    if not config:
        raise ModelNotFoundError(f"Model {model_name} not found in configuration")

    if config["requires_api_key"]:
        api_key = os.getenv(config["env_key"])
        if not api_key:
            raise ModelConfigurationError(
                f"API key {config['env_key']} required for {model_name} not found in environment variables"
            )


class ModelManager:
    def __init__(self):
        self.full_model_name = load_model_config()
        self.provider, self.model_name = parse_model_name(self, self.full_model_name)
        self._client: Optional[
            Union[ollama.AsyncClient, AsyncAnthropic, AsyncOpenAI]
        ] = None
        validate_api_keys(self.full_model_name)

    @property
    def client(self) -> Union[ollama.AsyncClient, AsyncAnthropic, AsyncOpenAI]:
        """Lazy initialization of client with proper typing"""
        if self._client is None:
            self._client = self._initialize_client()
        return self._client

    def _initialize_client(
        self,
    ) -> Union[ollama.AsyncClient, AsyncAnthropic, AsyncOpenAI]:
        try:
            if self.provider == ModelProvider.OLLAMA:
                return ollama.AsyncClient()
            elif self.provider == ModelProvider.ANTHROPIC:
                return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            elif self.provider == ModelProvider.OPENAI:
                return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            else:
                raise ModelConfigurationError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            raise ModelConfigurationError(
                f"Error initializing client for {self.provider}: {str(e)}"
            )

    async def _ollama_chat(self, messages: List[Dict], tools: Optional[List]) -> Dict:
        """Gestion spécifique Ollama avec typage correct"""
        client = cast(ollama.AsyncClient, self.client)
        response = await client.chat(
            model=self.model_name,
            messages=[
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ],
            # tools=tools if tools and MODEL_CONFIG[self.full_model_name]["supports_tools"] else None,
        )
        return self._format_ollama_response(response)

    async def _anthropic_chat(
        self, messages: List[Dict], tools: Optional[List]
    ) -> Dict:
        """Gestion spécifique Anthropic avec typage correct"""
        client = cast(AsyncAnthropic, self.client)
        response = await client.messages.create(
            model=self.model_name,
            max_tokens=MODEL_CONFIG[self.full_model_name]["max_tokens"],
            messages=[
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ],
            # tools=tools if tools and MODEL_CONFIG[self.full_model_name]["supports_tools"] else None,
        )
        return self._format_anthropic_response(response)

    async def _openai_chat(self, messages: List[Dict], tools: Optional[List]) -> Dict:
        """Gestion spécifique OpenAI avec typage correct"""
        client = cast(AsyncOpenAI, self.client)
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ],
            # tools=tools if tools and MODEL_CONFIG[self.full_model_name]["supports_tools"] else None,
        )
        return self._format_openai_response(response)

    async def chat_completion(
        self, messages: List[Dict[str, str]], tools: Optional[List] = None
    ) -> Dict[str, Any]:
        """Interface unifiée pour les chat completions"""
        try:
            formatted_messages = self._format_messages(messages)

            if self.provider == ModelProvider.OLLAMA:
                return await self._ollama_chat(formatted_messages, tools)
            elif self.provider == ModelProvider.ANTHROPIC:
                return await self._anthropic_chat(formatted_messages, tools)
            elif self.provider == ModelProvider.OPENAI:
                return await self._openai_chat(formatted_messages, tools)
            else:
                raise ModelConfigurationError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            raise ModelAPIError(
                f"Error in chat completion with {self.provider}: {str(e)}"
            )

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Formate les messages selon le provider
        """
        if self.provider == ModelProvider.ANTHROPIC:
            return [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]
        return messages

    def _format_ollama_response(self, response: Any) -> Dict:
        """Formate la réponse Ollama"""
        return {
            "content": response.get("message", {}).get("content", ""),
            "tool_calls": response.get("message", {}).get("tool_calls", []),
            "usage": {"total_tokens": 0},
        }

    def _format_anthropic_response(self, response: Any) -> Dict:
        """Formate la réponse Anthropic"""
        return {
            "content": response.content[0].text if response.content else "",
            "tool_calls": (
                response.tool_calls if hasattr(response, "tool_calls") else []
            ),
            "usage": {
                "total_tokens": (
                    (response.usage.input_tokens + response.usage.output_tokens)
                    if hasattr(response, "usage")
                    else 0
                )
            },
        }

    def _format_openai_response(self, response: Any) -> Dict:
        """Formate la réponse OpenAI"""
        return {
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls or [],
            "usage": (
                response.usage.dict()
                if hasattr(response, "usage")
                else {"total_tokens": 0}
            ),
        }
