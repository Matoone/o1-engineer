### FILE: agents/base_agent.py
from logger import logger

from typing import Dict, List, Optional, Any, Union
from model_manager import ModelManager, ModelError, ModelConfigurationError
from dotenv import load_dotenv


class BaseAgent:
    """
    Base class for all AI agents in the system.
    Handles common functionality like model initialization and basic chat interactions.
    """

    def __init__(self, model_kind: str = "DEFAULT_MODEL"):
        """
        Initialize the base agent with a specific model configuration.

        Args:
            model_kind (str): Environment variable key for the model name
        """
        load_dotenv()
        self.model_kind = model_kind
        self._model_manager: ModelManager = None  # type: ignore
        self.setup_model(model_kind)

    def setup_model(self, model_kind: str) -> None:
        """
        Set up the model manager with the specified model from environment variables.
        """
        try:

            # Initialize the model manager
            self._model_manager = ModelManager(model_kind)
            logger.info(
                f"Successfully initialized {self.__class__.__name__}. model kind: {model_kind} model name: {self._model_manager.full_model_name}"
            )

        except (ModelConfigurationError, ModelError) as e:
            logger.error(f"Error initializing model for {self.__class__.__name__}: {e}")
            raise

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        added_files: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Send a chat completion request to the AI model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            added_files: Optional dictionary of files to include in the context

        Returns:
            Optional[str]: The AI's response content or None if there's an error
        """
        try:
            if added_files:
                file_context = "Added files:\n"
                for file_path, content in added_files.items():
                    file_context += f"File: {file_path}\nContent:\n{content}\n\n"

                # Update the user's message with file context
                messages[-1]["content"] = f"{file_context}\n{messages[-1]['content']}"
                logger.info(f"Number of files in context: {len(added_files)}")
            # Log total size of messages in KB
            total_size = sum(len(str(msg).encode("utf-8")) for msg in messages) / 1024
            logger.info(f"Total messages size: {total_size:.2f} KB")

            response = await self._model_manager.chat_completion(messages=messages)  # type: ignore
            return response.get("content")

        except Exception as e:
            logger.error(f"Error in chat completion for {self.__class__.__name__}: {e}")
            return None
