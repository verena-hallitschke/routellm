"""Module for categorical base task in RouteLLM verbalization."""

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


class CategoricalBaseTask(BaseTask):
    """Categorical base task."""

    TOKEN_REASON_REPLACEMENT_TEXT = "<token_reason>"
    FEATURE_NAME = None
    CATEGORIES = []
    FILTER_LIST = []

    COT_CHANCE = 0.2
    POSITION_CHANCE = 0.3

    @abstractmethod
    def _get_general_task(
        self,
    ) -> str:
        """
        Get a general task description for the categorical feature.

        Returns:
            str: A randomly selected task description related to the categorical \
                feature.

        """
        pass

    @abstractmethod
    def _get_specific_value_task(self, target_value: str) -> str:
        """
        Get a specific task description for the categorical feature based on the target.

        Args:
            target_value (str): The target value for the categorical feature.

        Returns:
            str: A randomly selected task description related to the specific \
                categorical value.

        """
        pass

    @abstractmethod
    def _get_specific_response_true_single(
        self, segments: pd.DataFrame, target_value: str, use_cot: bool
    ) -> str:
        """
        Generate a response for a single segment with the target value.

        Args:
            segments (pd.DataFrame): DataFrame containing the route segments with \
                categorical feature values.
            target_value (str): The target value for the categorical feature.
            use_cot (bool): Whether to include a reasoning step in the response.

        Returns:
            str: A randomly selected response template indicating the presence of the \
                target value.

        """
        pass

    @abstractmethod
    def _get_specific_response_true_multi(
        self,
        segments: pd.DataFrame,
        target_value: str,
        use_cot: bool,
        position: str | None = None,
        plural: bool = False,
    ) -> str:
        """
        Generate a response for multiple segments with the target value.

        Args:
            segments (pd.DataFrame): DataFrame containing the route segments with \
                categorical feature values.
            target_value (str): The target value for the categorical feature.
            use_cot (bool): Whether to include a reasoning step in the response.
            position (str | None, optional): Position of the target value along the \
                route, if applicable. Defaults to None.
            plural (bool, optional): Indicates if there are multiple segments with the \
                target value. Defaults to False.

        Returns:
            str: A randomly selected response template indicating the presence of the \
                target value across multiple segments.

        """
        pass

    @abstractmethod
    def _get_specific_response_false(
        self, segments: pd.DataFrame, target_value: str, use_cot: bool
    ) -> str:
        """
        Generate a response for segments without the target value.

        Args:
            segments (pd.DataFrame): DataFrame containing the route segments with \
                categorical feature values.
            target_value (str): The target value for the categorical feature.
            use_cot (bool): Whether to include a reasoning step in the response.

        Returns:
            str: A randomly selected response template indicating the absence of the \
                target value.

        """
        pass

    @abstractmethod
    def _categories_to_text(
        self, unique_categories: list[str], position: str | None = None
    ) -> str:
        """
        Convert unique categories of the categorical feature into a descriptive text.

        Args:
            unique_categories (list[str]): List of unique categories for the \
                categorical feature.
            position (str | None, optional): Position of the categories along the \
                route, if applicable. Defaults to None.

        Returns:
            str: A descriptive text summarizing the unique categories.

        """
        pass

    def pre_process_df(self, route_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process the route DataFrame for the categorical task.

        Args:
            route_df (pd.DataFrame): DataFrame containing the route data with columns \
                for latitude and longitude.

        Returns:
            pd.DataFrame: Pre-processed DataFrame with relevant features for the \
                categorical task.

        """
        return route_df

    def get_task(
        self, unique_values: list[str]
    ) -> tuple[str, str | None, str, bool, bool]:
        """
        Get a task description based on the unique values of the categorical feature.

        Args:
            unique_values (list[str]): List of unique values for the categorical \
                feature.

        Returns:
            tuple[str, str | None, str, bool, bool]: A tuple containing the task text, \
                target value (if any), task type, whether to use chain-of-thought \
                (use_cot), and whether to return position.

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
        filtered_categories = [
            val for val in self.CATEGORIES if val not in self.FILTER_LIST
        ]
        task_type = random.choice(
            ["general", *["specific" for _ in range(len(filtered_categories))]]
        )

        use_cot = random.random() < self.COT_CHANCE
        return_position = random.random() < self.POSITION_CHANCE

        if task_type == "general":
            task_text = self._get_general_task()
            target_value = None
        else:
            unique_values = [
                val for val in unique_values if val not in self.FILTER_LIST
            ]
            if (
                unique_values is not None
                and len(unique_values) > 0
                and random.random() < 0.5
            ):
                target_value = random.choice(unique_values)
            else:
                target_value = random.choice(filtered_categories)

            task_text = self._get_specific_value_task(target_value)
            if use_cot and return_position:
                task_text += (
                    random.choice(["?", ".", "!", "?\n", ".\n", "!\n", "\n", "\n\n"])
                    + " "
                    + random.choice(
                        [
                            "Please explain your decision and say where "
                            + "along the route",
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
            target_value,
            task_type,
            use_cot,
            return_position,
        )

    def get_system_message(
        self, style_info: dict[str, str] | None = None
    ) -> tuple[str, dict[str, Any]]:
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

    def _general_generate_task(
        self,
        segments: pd.DataFrame,
    ) -> str:
        """
        Generate a general task description for the categorical feature.

        Args:
            segments (pd.DataFrame): DataFrame containing the route segments with \
                categorical feature values.

        Returns:
            str: A randomly selected task description related to the categorical \
                feature.

        """
        unique_values = segments[self.FEATURE_NAME].unique().tolist()

        return self._categories_to_text(unique_values)

    def _specific_generate_task(
        self,
        segments: pd.DataFrame,
        target_value: str,
        has_multiple_segments: bool,
        return_position: bool,
        use_cot: bool,
    ) -> str:
        """
        Generate a specific task description for the feature based on the target value.

        Args:
            segments (pd.DataFrame): DataFrame containing the route segments with \
                categorical feature values.
            target_value (str): The target value for the categorical feature.
            has_multiple_segments (bool): Indicates if there are multiple segments \
                with the target value.
            return_position (bool): Indicates if the position of the target value \
                should be included in the response.
            use_cot (bool): Indicates if the response should include a reasoning step.

        Returns:
            str: A randomly selected task description related to the specific \
                categorical value.

        """
        has_feature = (segments[self.FEATURE_NAME] == target_value).any()
        if not has_multiple_segments or not has_feature:
            return (
                self._get_specific_response_true_single(segments, target_value, use_cot)
                if has_feature
                else self._get_specific_response_false(segments, target_value, use_cot)
            )

        # At least one closure
        travel_time_lengths = (
            segments.groupby("segment")["current_travel_time"].sum().cumsum()
        )

        # Define position as start = 1/3, middle = 2/3 and end = 3/3
        route_start_mid = 1 / 3 * segments["current_travel_time"].sum()
        route_mid_end = 2 / 3 * segments["current_travel_time"].sum()

        target_indices = np.where(
            segments[segments["start_segment"]][self.FEATURE_NAME] == target_value
        )[0]

        plural = True

        # Check if only one segment has the feature
        if len(target_indices) == 1:
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

        min_feature_index = target_indices.min()
        max_feature_index = target_indices.max()
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

        return self._get_specific_response_true_multi(
            segments,
            target_value,
            use_cot,
            position=position if return_position else None,
            plural=plural,
        )

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
            task_text, target_value, task_type, use_cot, return_position = (
                self.get_task(
                    self.pre_process_df(splice_df.copy())[self.FEATURE_NAME]
                    .unique()
                    .tolist()
                )
            )

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
                splice_df = self.pre_process_df(splice_df)
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
                segments["start_segment"] = False
                segments.loc[similar_row_index, "start_segment"] = True

                if task_type == "specific":
                    has_multiple_segments = len(segments[similar_row_index]) > 1

                    route_text = self._specific_generate_task(
                        segments,
                        target_value,
                        has_multiple_segments,
                        return_position,
                        use_cot,
                    )
                else:
                    route_text = self._general_generate_task(segments)

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
                    "target_value": target_value,
                    "task_type": task_type,
                    "has_feature": None
                    if task_type == "general"
                    else bool((splice_df[self.FEATURE_NAME] == target_value).any()),
                    "use_cot": use_cot,
                    "feature_indices": np.argwhere(
                        (splice_df[self.FEATURE_NAME] == target_value).to_numpy()
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

            response = response.replace(
                CategoricalBaseTask.TOKEN_REASON_REPLACEMENT_TEXT, token_text
            )

        return task["task_text"], response
