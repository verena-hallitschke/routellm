"""Module for the base Chain-of-Thought (COT) task in RouteLLM."""

import random
from typing import Any

import pandas as pd

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask
from routellm.dataset.verbalization.text_template import route2text


class COTBaseTask(BaseTask):
    """Base class for Chain-of-Thought (COT) tasks in RouteLLM."""

    MIN_SEQUENCE_LENGTH = 20
    MAX_ROUTE_TOKENS = 150
    NON_TEXT_COLUMNS = []
    MAX_NUM_REQUESTS = 5  # Only query this number of sequences per route

    def pre_process_df(self, route_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process the route DataFrame for the COT task.

        Args:
            route_df (pd.DataFrame): DataFrame containing the route data with columns \
                for latitude and longitude.

        Returns:
            pd.DataFrame: Pre-processed DataFrame with relevant features for the COT \
                task.

        """
        return route_df

    def convert_to_text(self, route_df: pd.DataFrame) -> str | None:
        """
        Convert the route DataFrame to a textual description.

        Args:
            route_df (pd.DataFrame): DataFrame containing the route data with columns \
                for latitude and longitude.

        Returns:
            str | None: A textual description of the route, or None if conversion \
                fails.

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

        Keep the language diverse. Make the answer sound how humans would answer to the question. Answer the question, do not describe the route!

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

    def get_task(self) -> str:
        """
        Get a random task from the predefined task list.

        Returns:
            str: Task text selected from the task list.

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

        cot_list = [
            "Answer the question step by step to make it understandable how you came "
            + "to your conclusion. ",
            "Give a step by step rundown. ",
            "Please answer step by step. ",
            "Please explain how you reached the conclusion step by step. ",
        ]

        prepend_list = [
            "",
            "Please answer in full sentences. ",
            "Please give a short an concise answer. ",
            "",
            "Answer in full sentences. ",
            "Be concise. ",
            "Short answer. ",
            "In short. ",
            "In very few sentences. ",
            "Explain to a child. ",
            "Explain to a customer. ",
            "Answer to a friend. ",
            "Give a long answer. ",
            "Give an answer in one sentence only. ",
            "Give an answer in every-day language. ",
            "Keeping in mind that this person is also driving a sports car. ",
            "Make sure to use a beautiful language. ",
        ]
        cot_string = random.choice(
            cot_list
            if random.random() < 0.5
            else prepend_list
            if random.random() < 0.3
            else [""]
        )

        return (
            cot_string
            + random.choice(self.TASK_LIST)
            + random.choice(punctuation_characters)
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

        Raises:
            NotImplementedError: If the task does not use LLMs for answering.

        Returns:
            list[dict[str, Any]]: List of generated tasks, each containing the task \
                description and model response.

        """
        if not self.use_llm_for_answering:
            raise ValueError(
                "This task needs llm postprocessing (use_llm_for_answering is set "
                + "to False)!"
            )

        cleaned_df = clean_gdf(route_df.copy())

        if cleaned_df is None:
            return None

        cleaned_df = self.pre_process_df(cleaned_df)

        splice_list = []

        total_splice_length = 0

        # Split into random length splices
        if self.allow_splicing:
            while total_splice_length < len(cleaned_df):
                if (
                    len(cleaned_df) - total_splice_length < self.MIN_SEQUENCE_LENGTH * 2
                    and len(cleaned_df) - total_splice_length <= self.MAX_ROUTE_TOKENS
                ):
                    if (
                        len(cleaned_df) - total_splice_length
                        >= self.MIN_SEQUENCE_LENGTH
                    ):
                        splice_list.append(
                            list(range(total_splice_length, len(cleaned_df)))
                        )

                    break

                # Random set_back for overlap

                set_back = random.randint(0, int(0.5 * self.MIN_SEQUENCE_LENGTH))
                total_splice_length = max(total_splice_length - set_back, 0)

                length = random.randint(self.MIN_SEQUENCE_LENGTH, self.MAX_ROUTE_TOKENS)
                splice_list.append(
                    list(
                        range(
                            total_splice_length,
                            min(total_splice_length + length, len(cleaned_df)),
                        )
                    )
                )
                total_splice_length += length
        else:
            splice_list = [list(range(len(cleaned_df)))]

        task_list = []

        # drop random routes
        random.shuffle(splice_list)
        splice_list = splice_list[: min(len(splice_list), self.MAX_NUM_REQUESTS)]

        for splice in splice_list:
            splice_df = cleaned_df.iloc[splice]
            system_message, style = self.get_system_message(style_info=style)
            task_text = self.get_task()

            textual_description = self.convert_to_text(splice_df)
            message = f"{task_text}"
            message += f"\n{textual_description}"

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
            }
            task_list.append(self.process_task(task_description, history=history))

        return task_list
