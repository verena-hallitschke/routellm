"""Module defining the BooleanBaseTask class for boolean tasks."""

import random
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask
from routellm.dataset.verbalization.text_template import splice_route


class BooleanBaseTask(BaseTask):
    """Base class for boolean tasks in RouteLLM verbalization."""

    ONE_SEGMENT_TRUE_TEMPLATE = []
    FALSE_TEMPLATE = []
    FEATURE_NAME = None
    COT_CHANCE = 0.2
    POSITION_CHANCE = 0.3

    @abstractmethod
    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ) -> str:
        """
        Generate a multi-segment response based on the task's feature presence.

        Args:
            plural (bool): Indicates if the response should be plural (present in \
                multiple segments).
            return_position (bool): Indicates if the position of the feature should be \
                included in the response.
            position (str): The position of the feature in the route, if applicable.

        Raises:
            NotImplementedError: This is an abstract method and should be implemented \
                in subclasses.

        Returns:
            str: A randomly selected response template indicating the presence of the \
                feature.

        """
        raise NotImplementedError(
            "This method should be implemented in subclasses to provide multi-segment \
responses."
        )
        pass

    def get_task(self) -> tuple[str, bool, bool]:
        """
        Get a random task from the predefined task list.

        Returns:
            tuple[str, bool, bool]: A tuple containing the task text, a boolean \
                indicating if chain-of-thought reasoning is used, and a boolean \
                indicating if the position along the route should be returned.

        """
        punctuation_characters = [
            "?",
            ".",
            ":",
            "",
            " ",
            "\n",
            "\n\n",
        ]

        use_cot = random.random() < self.COT_CHANCE
        return_position = random.random() < self.POSITION_CHANCE
        task_text = random.choice(self.TASK_LIST)

        if use_cot and return_position:
            task_text += (
                random.choice(["?", ".", "!", "?\n", ".\n", "!\n", "\n", "\n\n"])
                + " "
                + random.choice(
                    [
                        "Please explain your decision and say where along the route",
                        "Why and when",
                        "Explain why and when",
                        "Add why and when",
                        "Please explain why and add the position along the route",
                        "Provide the reason why and mention when",
                    ]
                )
            )
        elif use_cot:
            task_text += random.choice(
                ["? ", ". ", "! ", "?\n", ".\n", "!\n", "\n", "\n\n"]
            ) + random.choice(
                [
                    "Please explain your decision",
                    "Provide reasons for your answer",
                    "In addition to the answer, explain why it is correct",
                    "Explain your answer",
                    "Please explain",
                ]
            )

        elif return_position:
            task_text += (
                random.choice(["?", ".", "!", "?\n", ".\n", "!\n", "\n", "\n\n"])
                + " "
                + random.choice(
                    [
                        "Where along the route",
                        "Also answer where in the route",
                        "Please add where",
                        "Where",
                        "When",
                        "When along the route",
                    ]
                )
            )

        return (
            task_text + random.choice(punctuation_characters),
            use_cot,
            return_position,
        )

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
        system_message = """You are an AI assistant that helps people find information about routes.

        Keep the language diverse. Make the answer sound how humans would answer to the question. Answer the question, do not describe the route! Give short and precise answers. If not prompted differently, answer in one or 2 sentences!

        The answers you give will be used to create an instruction dataset for a multimodal large language model. This model will get special tokens as the routes whereas you get a textual description.
        The textual description is generated with a template and based on feature values from osm, the tomtom traffic api and the azure maps weather api. Use this information as additional knowledge but do not directly include it in your answer.
        Here is a short explanation of every feature in the description:

        length: length of the osm segment
        grade: grade in percent calculated from the OSMNX graph and an elevation map
        curvature: menger curvature (1 / curve radius) calculated from osmnx graph
        heading: cardinal direction calculated from the osmnx graph
        landuse: Describes what the area right next to the street is used for. Based on the osm landuse tag but was grouped into new categories
        free flow speed: Free flow speed calculated using the tomtom api
        oneway: Whether the osm segment is a oneway street
        lanes: number of lanes (osm data)
        highway: type of street (converted to text) based on osm street types
        speed limit: based on the osm speed_kph tag
        variable speed: based on the osm maxspeed tag. Describes whether the speed limit on a segment could change based on a sign or a traffic signal.
        bridge: Whether the segment contains a bridge
        tunnel: whether the segment contains a tunnel
        junction: Describes whether this segment is part of a junction/crossing
        traffic incident information: Based on the tomtom traffic api and describes the type of incident, how certain it is that it appears, the magnitude of the incident and the expected delay caused by the incident
        weather information: From the azure maps weather api. Contains precipitation, percentage of cloud coverage, wind speed, wind gust speed and whether there is lightning.

        Use the textual description for your answer but do not directly refer to phrasing or the text in your answer. Instead talk about the route/journey/trip/segments/steps/etc.
        """

        if style_info is None:
            style_text, style = self.get_random_style()
        else:
            style = style_info
            style_text = get_style(**style_info)

        return (
            f"{system_message} Follow these phrasing instructions: {style_text}",
            style,
        )  # f"{system_message} Follow these style instructions: {style_text}", style

    def _df_generate_task(
        self, segments: pd.DataFrame, has_multiple_segments: bool, return_position: bool
    ) -> str:
        """
        Generate a task response based on the segments DataFrame.

        Args:
            segments (pd.DataFrame): DataFrame containing route segments with features.
            has_multiple_segments (bool): Indicates if the route has multiple \
                relevant segments.
            return_position (bool): Indicates if the position of the feature should \
                be included in the response

        Returns:
            str: A randomly selected response template indicating the presence of the \
                feature.

        """
        has_feature = segments[self.FEATURE_NAME].any()
        if not has_multiple_segments or not has_feature:
            return random.choice(
                self.ONE_SEGMENT_TRUE_TEMPLATE if has_feature else self.FALSE_TEMPLATE
            )

        # At least one closure
        travel_time_lengths = (
            segments.groupby("segment")["current_travel_time"].sum().cumsum()
        )

        # Define position as start = 1/3, middle = 2/3 and end = 3/3
        route_start_mid = 1 / 3 * segments["current_travel_time"].sum()
        route_mid_end = 2 / 3 * segments["current_travel_time"].sum()

        closure_indices = np.where(
            segments.groupby("segment")[self.FEATURE_NAME].any()
        )[0]

        plural = True

        # Check if only one segment has the feature
        if len(closure_indices) == 1:
            plural = False

        journey_word = random.choice(
            [
                "of our ongoing trip",
                "of the trip",
                "of the route",
                "of the path",
                "of our path",
                "of your route",
                "of the journey",
                "of the provided route",
                "of your travel",
                "of this specific path",
                "of the way",
                "of the course",
                "of the itinerary",
            ]
        )

        position = "throughout the journey"

        min_feature_index = closure_indices.min()
        max_feature_index = closure_indices.max()
        if travel_time_lengths.iloc[max_feature_index] < route_start_mid:
            position = f"in the beginning {journey_word}"
        elif (
            route_start_mid
            <= travel_time_lengths.iloc[min_feature_index]
            <= travel_time_lengths.iloc[max_feature_index]
            < route_mid_end
        ):
            position = f"in the middle {journey_word}"
        elif route_mid_end <= travel_time_lengths.iloc[min_feature_index]:
            position = f"in the end {journey_word}"

        return self._get_multi_segment_response(plural, return_position, position)

    def generate_task(
        self,
        route_df: pd.DataFrame,
        max_text_length: int = 3 * 29000,  # noqa: ARG002
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

        Returns:
            list[dict[str, Any]]: List of generated tasks, each containing the task \
                description and model response.

        """
        cleaned_df = clean_gdf(route_df.copy())

        if cleaned_df is None:
            return None

        splice_list = [list(range(len(cleaned_df)))]
        if len(cleaned_df) > self.MAX_ROUTE_TOKENS and self.allow_splicing:
            splice_list = splice_route(len(cleaned_df), self.MAX_ROUTE_TOKENS)

        task_list = []
        for splice in splice_list:
            splice_df = cleaned_df.iloc[splice]
            system_message, style = self.get_system_message(style_info=style)
            task_text, use_cot, return_position = self.get_task()

            if self.use_llm_for_answering:
                textual_description = self.convert_to_text(splice_df)

                message = f"{task_text}"
                message += f"\n{textual_description}"

                if use_cot:
                    message = (
                        "Do it step by step. Please provide reasons for your answers. "
                        + message
                    )
            else:
                selection_columns = [self.FEATURE_NAME]
                segments = splice_df.copy()
                similar_row_index = ~splice_df[selection_columns].eq(
                    splice_df[selection_columns].shift()
                ).all(axis=1)

                segments["segment"] = np.nan
                segments.loc[similar_row_index, "segment"] = np.arange(
                    len(similar_row_index[similar_row_index])
                )
                segments["segment"] = segments["segment"].ffill().astype("int64")

                has_multiple_segments = len(segments[similar_row_index]) > 1

                route_text = self._df_generate_task(
                    segments, has_multiple_segments, return_position
                )
                system_message += f"Original task: {task_text}."
                message = f"{route_text}"

            task_description = {
                "system": system_message,
                "message": message,
                "style": style,
                "segments": None,
                "header_text": None,
                "address_header_text": None,
                "segment_texts": None,
                "task_text": task_text,
                "include_header": False,
                "print_step_id": False,
                "use_orig_steps": False,
                "splice_indices": splice_df["step"].to_list(),
                "splice_arrays": True,
                "raw_route": None,
                "task_name": self.NAME,
                "task_specific": {
                    "has_feature": bool(splice_df[self.FEATURE_NAME].any()),
                    "use_cot": use_cot,
                    "feature_indices": np.argwhere(
                        splice_df[self.FEATURE_NAME].to_numpy()
                    )
                    .flatten()
                    .tolist(),
                    "return_position": return_position,
                },
            }
            task_list.append(self.process_task(task_description, history=history))

        return task_list

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

        if (
            task["task_specific"]["use_cot"]
            and len(task["task_specific"]["use_cot"]) > 0
        ):
            indicator_tokens = list(
                set(
                    [
                        tokenization[tok]
                        for tok in task["task_specific"]["feature_indices"]
                    ]
                )
            )
            route_tokens = [
                route_formatting_callback([tok], add_special_tokens=False)
                for tok in indicator_tokens
            ]
            token_text = (
                ", ".join(route_tokens[:-1]) + f", and {route_tokens[-1]}"
                if len(route_tokens) > 1
                else route_tokens[0]
            )

            response += random.choice(
                [
                    f" This is indicated by the following tokens: {token_text}.",
                    f" Please refer to the following tokens: {token_text}.",
                    f" Indicators for are: {token_text}.",
                    f" This is signified by: {token_text}.",
                ]
            )

        return task["task_text"], response
