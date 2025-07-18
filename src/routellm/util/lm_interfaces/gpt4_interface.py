"""Module defining an interface to the GPT API."""

from typing import Generator

import openai
from openai._types import NOT_GIVEN  # API does not accept none

from routellm.util.config import Config
from routellm.util.lm_interfaces import DEFAULT_SYSTEM_MESSAGE, GenericLMInterface


class GPTInterface(GenericLMInterface):
    """Interface for the GPT language model."""

    def __init__(
        self,
        default_model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: str | None = None,
    ):
        """
        Initialize a GPT interface with default parameters.

        Args:
            default_model (str, optional): Default model to use. Defaults to "gpt-4".
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
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        config = Config()

        self.client_standard = openai.AzureOpenAI(
            api_key=config.get("openai-api-key"),
            api_version="2023-07-01-preview",
            azure_endpoint=config.get("openai-instance"),
        )
        self.client_turbo = openai.AzureOpenAI(
            api_key=config.get("gptvision_openai-api-key"),
            api_version="2023-07-01-preview",
            azure_endpoint=config.get("gptvision_openai-instance"),
        )

        self.default_model = default_model

        self.available_models = {
            "gpt4-32k": 32768,
            "gpt-4": 8192,
            "gpt-35-turbo": 4000,
            "gpt-35-turbo-613": 4000,
            "gpt-4-vision": 128000,
            "gpt-4o": 128000,
        }

        self._new_ressource_models = ["gpt-4-vision", "gpt-4o"]

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
        force_model: bool = False,
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
            history (list[dict[str, str]] | None, optional): Chat history. Defaults \
                to None.
            force_model (bool, optional): If True, forces the use of the specified \
                model even if it is not suitable for the input size. Defaults to False.

        Returns:
            str: The generated completion from the model.

        """
        if history is None:
            history = []

        full_input = [
            *history,
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]

        tokens_to_generate = max_tokens or self.max_tokens
        token_count = tokens_to_generate + sum(
            [len(input_message["content"]) / 3.7 for input_message in full_input]
        )

        model_to_use = model_name if model_name is not None else self.default_model
        if model_to_use == "gpt4-32k" and token_count < 8000 and not force_model:
            # Use smaller model
            model_to_use = "gpt-4"

        temperature = temperature or self.temperature or NOT_GIVEN
        top_p = top_p or self.top_p or NOT_GIVEN
        frequency_penalty = frequency_penalty or self.frequency_penalty or NOT_GIVEN
        presence_penalty = presence_penalty or self.presence_penalty or NOT_GIVEN
        stop = stop or self.stop or NOT_GIVEN

        if model_to_use in self._new_ressource_models:
            client = self.client_turbo
        else:
            client = self.client_standard

        response = client.chat.completions.create(
            model=model_to_use,
            messages=full_input,
            temperature=temperature,
            max_tokens=tokens_to_generate,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )

        return response.choices[0].message.content

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
        force_model: bool = False,
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
            history (list[dict[str, str]] | None, optional): Chat history. Defaults \
                to None.
            force_model (bool, optional): If True, forces the use of the specified \
                model even if it is not suitable for the input size. Defaults to False.

        Yields:
            str: The generated text from the model, streamed in chunks.

        """
        if history is None:
            history = []
        full_input = [
            *history,
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]

        tokens_to_generate = max_tokens or self.max_tokens
        token_count = tokens_to_generate + sum(
            [len(input_message["content"]) / 3.7 for input_message in full_input]
        )

        model_to_use = model_name if model_name is not None else self.default_model
        if model_to_use == "gpt4-32k" and token_count < 8000 and not force_model:
            # Use smaller model
            model_to_use = "gpt-4"

        temperature = temperature or self.temperature or NOT_GIVEN
        top_p = top_p or self.top_p or NOT_GIVEN
        frequency_penalty = frequency_penalty or self.frequency_penalty or NOT_GIVEN
        presence_penalty = presence_penalty or self.presence_penalty or NOT_GIVEN
        stop = stop or self.stop or NOT_GIVEN

        if model_to_use in self._new_ressource_models:
            client = self.client_turbo
        else:
            client = self.client_standard

        response = client.chat.completions.create(
            model=model_to_use,
            messages=full_input,
            temperature=temperature,
            max_tokens=tokens_to_generate,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=True,
        )

        output_message = ""
        for chunk in response:
            if len(chunk.choices) > 0:
                output_message += chunk.choices[0].delta.content or ""

            yield output_message
