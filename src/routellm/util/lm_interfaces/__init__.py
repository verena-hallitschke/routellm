"""Module for defining the generic interface for language models (LMs) in RouteLLM."""

from abc import ABC, abstractmethod
from typing import Generator

DEFAULT_SYSTEM_MESSAGE = (
    "You are an AI assistant that helps people find information."
)


class GenericLMInterface(ABC):
    """Generic interface for language models in RouteLLM."""

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int=2000,
        top_p: float=0.95,
        frequency_penalty: float=0,
        presence_penalty: float=0,
        stop: str | None = None,
    ):
        """
        Initialize a language model interface with default parameters.

        Args:
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of generated tokens. Defaults \
                to 2000.
            top_p (float, optional): Generation top p. Defaults to 0.95.
            frequency_penalty (float, optional): Generation frequency penalty. \
                Defaults to 0.
            presence_penalty (float, optional): Generation presence penalty. \
                Defaults to 0.
            stop (str, optional): Tokens to stop the generation at. Defaults to None.

        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

        self.available_models = {}

    @abstractmethod
    def complete(
        self,
        message: str,
        model_name: str | None = None,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a completion for the given message using the language model.

        Args:
            message (str): Message to complete.
            model_name (str | None, optional): Name of the model. If None use instance \
                default. Defaults to None.
            system_message (str, optional): System Message. Defaults to "You are an AI \
                assistant that helps people find information.".
            temperature (float | None, optional): Sampling temperature. If None use \
                instance default. Defaults to None.
            max_tokens (int | None, optional): Maximum number of tokens. If None use \
                instance default. Defaults to None.
            top_p (float | None, optional): Top p. If None use instance default. \
                Defaults to None.
            frequency_penalty (float | None, optional): Frequency penalty. If None use \
                instance default. Defaults to None.
            presence_penalty (float | None, optional): Presence penalty. If None use \
                instance default. Defaults to None.
            stop (str | None, optional): Stop sequence. If None use instance default. \
                Defaults to None.
            history (list[dict[str, str]] | None, optional): Chat history. Defaults to \
                None.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            str: The generated completion from the model.

        """
        pass

    @abstractmethod
    def stream_complete(
        self,
        message: str,
        model_name: str | None = None,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Generate and stream a completion for the given message using the language model.

        Args:
            message (str): Message to complete.
            model_name (str | None, optional): Name of the model. If None use instance \
                default. Defaults to None.
            system_message (str, optional): System Message. Defaults to "You are an AI \
                assistant that helps people find information.".
            temperature (float | None, optional): Sampling temperature. If None use \
                instance default. Defaults to None.
            max_tokens (int | None, optional): Maximum number of tokens. If None use \
                instance default. Defaults to None.
            top_p (float | None, optional): Top p. If None use instance default. \
                Defaults to None.
            frequency_penalty (float | None, optional): Frequency penalty. If None use \
                instance default. Defaults to None.
            presence_penalty (float | None, optional): Presence penalty. If None use \
                instance default. Defaults to None.
            stop (str | None, optional): Stop sequence. If None use instance default. \
                Defaults to None.
            history (list[dict[str, str]] | None, optional): Chat history. Defaults to \
                None.
            **kwargs: Additional keyword arguments for the model.

        Yields:
            str: The generated text from the model, streamed in chunks.

        """
        pass
