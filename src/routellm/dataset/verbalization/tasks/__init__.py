"""Module for defining and managing tasks in the RouteLLM verbalization system."""

import glob
import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib import import_module
from inspect import isclass
from pkgutil import iter_modules
from typing import Any, Self

import pandas as pd
from openai import BadRequestError, RateLimitError

from routellm.dataset.verbalization.styles import get_random_style, get_style
from routellm.dataset.verbalization.text_template import route2text
from routellm.util.lm_interfaces import GenericLMInterface


class BaseTask(ABC):
    """Abstract base class for tasks in the RouteLLM verbalization system."""

    NAME = "base"
    VERSION = 0
    TASK_LIST = []
    NON_TEXT_COLUMNS = []
    MAX_ROUTE_TOKENS = 2900
    SUPPORTS_LLM_POSTPROCESSING = True

    def __new__(
        cls,
        lm_interface: GenericLMInterface,  # noqa: ARG004
        use_llm_for_answering: bool = True,  # noqa: ARG004
        allow_splicing: bool = True,  # noqa: ARG004
    ) -> Self:
        """
        Create a new instance of the task.

        Args:
            lm_interface (GenericLMInterface): Interface for interacting with the \
                language model.
            use_llm_for_answering (bool, optional): Whether to use the LLM for \
                answering tasks. Defaults to True.
            allow_splicing (bool, optional): Whether to allow splicing of overflowing \
                routes. Defaults to True.

        Raises:
            NotImplementedError: If the task is an abstract base task and cannot be \
                instantiated.

        Returns:
            Self: An instance of the task class.

        """
        if cls.NAME == BaseTask.NAME:
            raise NotImplementedError(
                "This task is an abstract base task and cannot be instantiated. Add a \
new subclass and with a custom NAME property to create a new task."
            )
        return super().__new__(cls)

    def __init__(
        self,
        lm_interface: GenericLMInterface,
        use_llm_for_answering: bool = True,
        allow_splicing: bool = True,
    ):
        """
        Initialize the task.

        Args:
            lm_interface (GenericLMInterface): Interface for interacting with the \
                language model.
            use_llm_for_answering (bool, optional): Whether to use the LLM for \
                answering tasks. Defaults to True.
            allow_splicing (bool, optional): Whether to allow splicing of overflowing \
                routes. Defaults to True.

        """
        self.use_llm_for_answering = use_llm_for_answering
        self.lm_interface = lm_interface

        self.allow_splicing = allow_splicing

    def postprocessing(self, response: str, settings: dict[str, Any]) -> str:  # noqa: ARG002
        """
        Post-process the response from the language model.

        Args:
            response (str): Response from the language model.
            settings (dict[str, Any]): Task settings, including the task text \
                and style.

        Returns:
            str: Post-processed response text.

        """
        return response

    def get_random_style(self) -> tuple[str, dict[str, str]]:
        """
        Get a random style for the task.

        Returns:
            tuple[str, dict[str, str]]: Tuple containing the style text and \
                style information.

        """
        style = get_random_style()

        style_text = get_style(**style)

        return style_text, style

    def get_task(self) -> str:
        """
        Get a random task from the predefined task list.

        Returns:
            str: Task text selected from the task list.

        """
        punctuation_characters = [
            "? ",
            ". ",
            ": ",
            "",
            " ",
            "\n ",
            "\n\n ",
        ]
        return random.choice(self.TASK_LIST) + random.choice(punctuation_characters)

    def convert_to_text(self, route_df: pd.DataFrame) -> str | None:
        """
        Convert a route DataFrame to a textual description.

        Args:
            route_df (pd.DataFrame): DataFrame containing route data with columns \
                for latitude and longitude.

        Returns:
            str | None: Textual description of the route, or None if conversion fails.

        """
        description_result = route2text(
            route_df,
            user_ignore_columns=self.NON_TEXT_COLUMNS,
            preprocess_df=False,
            print_street_names=False,
            print_step_id=False,
        )

        if description_result is None:
            return None
        header, step_list, _, _ = description_result[0]

        return header + " ".join(step_list)

    @abstractmethod
    def get_system_message(
        self, style_info: dict[str, str] | None = None
    ) -> tuple[str, dict[str, str]]:
        """
        Get the system message for the task.

        Args:
            style_info (dict[str, str] | None, optional): Style description for the \
                task. Defaults to None.

        Returns:
            tuple[str, dict[str, str]]: Tuple containing the system message and style \
                information.

        """
        pass

    @staticmethod
    def resolve(
        task: dict[str, Any],
        response: str,
        tokenization: list[str],
        route_formatting_callback: Callable[[list[str], Any], str],
        add_full_route: bool = True,
        **format_kwargs,
    ) -> tuple[str, str]:
        """
        Insert the tokenization into the task and format the response.

        Args:
            task (dict[str, Any]): Task description containing the task text and style.
            response (str): Response from the language model.
            tokenization (list[str]): List of tokens to be inserted into the task.
            route_formatting_callback (Callable[[list[str], Any], str]): Callback \
                function to format the tokenized route.
            add_full_route (bool, optional): Whether to add the full route to the task.
                Defaults to True.
            **format_kwargs: Additional keyword arguments for formatting the route.

        Returns:
            tuple[str, str]: Tuple containing the formatted task text and response.

        """
        # Insert tokenization into task

        if add_full_route:
            task["task_text"] = task["task_text"] + route_formatting_callback(
                tokenization, **format_kwargs
            )

        return task["task_text"], response

    def process_task(
        self,
        task_description: dict[str, Any],
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any] | None:
        """
        Process the task description and generate a response.

        Args:
            task_description (dict[str, Any]): Task description containing the system \
                message, message, style, and other task-related information.
            history (list[dict[str, str]] | None, optional): Chat history for the \
                task. Defaults to None.

        Returns:
            dict[str, Any] | None: A dictionary containing the task description and \
                model response, or None if the task cannot be processed.

        """
        if self.use_llm_for_answering:
            success = False

            while not success:
                try:
                    lm_response = self.lm_interface.complete(
                        task_description["message"],
                        system_message=task_description["system"],
                        history=history,
                    )
                    success = True
                except RateLimitError:
                    time.sleep(20.0)
                    continue
                except BadRequestError:
                    return None
            model_response = self.postprocessing(lm_response, task_description)
        else:
            model_response = self.postprocessing(
                task_description["message"], task_description
            )

        return {"task": task_description, "response": model_response}

    def generate_task(
        self,
        route_df: pd.DataFrame,
        max_text_length: int = 3 * 29000,
        history: list[dict[str, str]] | None = None,
        style: dict[str, str] | None = None,
        route_path: str | None = None,  # noqa: ARG002
    ) -> list[dict[str, Any]] | None:
        """
        Generate a task based on the provided route DataFrame.

        Args:
            route_df (pd.DataFrame): DataFrame containing route data with columns \
                for latitude and longitude.
            max_text_length (int, optional): Maximum length of the generated text. \
                Defaults to 3 * 29000.
            history (list[dict[str, str]] | None, optional): Chat history for the \
                task. Defaults to None.
            style (dict[str, str] | None, optional): Style description for the task. \
                Defaults to None.
            route_path (str | None, optional): Path to the route data. Defaults to \
                None.

        Raises:
            NotImplementedError: If the task does not use LLMs for answering.

        Returns:
            list[dict[str, Any]]: List of generated tasks, each containing the task \
                description and model response.

        """
        if not self.use_llm_for_answering:
            raise NotImplementedError("Method without LLMs not implemented!")

        text_response = route2text(
            route_df,
            max_num_segments=234,
            buffer_segments_to=300,
            print_street_names=False,
            print_step_id=False,
            splice_overflowing_routes=self.allow_splicing,
        )

        if text_response is None:
            return []

        task_list = []

        for header_text, segment_texts, segment_map, splice_map in text_response:
            route_text = header_text
            route_text += "".join(segment_texts)

            if len(route_text) > max_text_length:
                continue

            task_text = self.get_task()
            task_sys_message, style = self.get_system_message(style_info=style)

            message = f"{task_text}"
            message += f"\n{route_text}"

            task_description = {
                "system": task_sys_message,
                "message": message,
                "style": style,
                "segments": segment_map,
                "header_text": header_text,
                "address_header_text": None,
                "segment_texts": segment_texts,
                "task_text": task_text,
                "include_header": False,
                "print_step_id": False,
                "use_orig_steps": False,
                "splice_indices": splice_map,
                "splice_arrays": True,
                "raw_route": route_text,
                "task_name": self.NAME,
            }

            task_list.append(self.process_task(task_description, history=history))
        return task_list


class TaskRegistry:
    """Registry for managing available tasks in the RouteLLM verbalization system."""

    # Singleton
    def __new__(cls) -> Self:
        """
        Create a new instance of the TaskRegistry.

        This class is a singleton, meaning only one instance will be created.

        Returns:
            Self: An instance of the TaskRegistry class.

        """
        if not hasattr(cls, "instance"):
            cls.instance = super(TaskRegistry, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        """Create the task registry and load all available tasks."""
        if not hasattr(self, "registry"):
            self.registry: dict[str, type[BaseTask]] = {}

            for file_finder, module_name, _ in iter_modules(
                [
                    os.path.dirname(__file__),
                    *glob.glob(os.path.join(os.path.dirname(__file__), "*/")),
                ]
            ):
                module_path = f"{__name__}"

                folder_name = (
                    os.path.basename(os.path.dirname(file_finder.path))
                    if file_finder.path[-1] == "/"
                    else os.path.basename(file_finder.path)
                )

                if folder_name != "tasks":
                    module_path += (
                        f".{os.path.basename(os.path.dirname(file_finder.path))}"
                    )

                try:
                    module = import_module(f"{module_path}.{module_name}")
                except NotImplementedError:
                    continue

                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)

                    if (
                        isclass(attribute)
                        and issubclass(attribute, BaseTask)
                        and attribute.NAME != BaseTask.NAME
                    ):
                        # Add the class to registry
                        self.registry[attribute.NAME] = attribute

    def __getitem__(self, name: str) -> type[BaseTask]:
        """
        Get a task class by its name.

        Args:
            name (str): Name of the task class to retrieve.

        Returns:
            type[BaseTask]: The task class associated with the given name.

        """
        return self.registry[name]

    def get_available_tasks(self) -> list[str]:
        """
        Get a list of all available task names in the registry.

        Returns:
            list[str]: List of names of all available tasks.

        """
        return list(self.registry.keys())
