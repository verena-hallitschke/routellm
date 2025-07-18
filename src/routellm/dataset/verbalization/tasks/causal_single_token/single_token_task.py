import random
import warnings
from typing import Any

import pandas as pd

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.routes.const import (
    convert_cloud_cover_to_word,
    convert_heading_to_direction,
    convert_precipitation_to_word,
    convert_wind_gust_diff_to_word,
    convert_wind_to_word,
)
from routellm.dataset.verbalization.const_to_text import (
    convert_highway_to_text,
    convert_incident_to_text,
    convert_landuse_to_text,
    convert_probability_to_text,
)
from routellm.dataset.verbalization.tasks import BaseTask


def feature_to_text(feature, value):
    text_map = {
        "lanes": "The number of lanes is {value}.",
        "speed_kph": "The speed limit is {value} kph.",
        "wind_speed": "The wind speed is {value} kph.",
        "incident_delay": "The incident delay is {value} s.",
        "cloudCover": "The cloud coverage is {value} %.",
        "windGust_speed": "The speed of the wind gusts is {value} kph.",
        "incident_distance": "The incident distance is {value} m.",
        "travel_time": "The travel time is {value} s.",
        "temperature": "The temperature is {value} degrees Celsius.",
        "precipitation": "The precipitation is {value} dBZ.",
        "grade_abs": "The absolute grade of the road is {value} %.",
        "grade": "The grade of the road is {value} %.",
        "free_flow_speed": "The free flow speed is {value} kph.",
        "current_speed": "The current travel speed is {value} kph.",
        "landuse_distance": "The surrounding area is {value} m away.",
        "length": "The length of the road is {value} m.",
        "curvature": "The curvature is {value} 1/m.",
        "bearing": "The heading of the segment is {value} degrees.",
        "current_travel_time": "The current travel time is {value} s.",
        "cos_time": "The cosine of the second in the day is {value}.",
        "sin_time": "The sine of the second in the day is {value}.",
        "delay": "The speed delay factor is {value}.",
    }

    binary_text_map = {
        "oneway": f"Oneway: This is {'' if value else 'not '}a oneway street.",
        "tunnel": f"Tunnel: There is {'' if value else 'no '}tunnel",
        "bridge": f"Bridge there is {'' if value else 'no '}bridge.",
        "flow_road_closure": f"Road closure: the road is {'' if value else 'not '}closed.",
        "weekday": f"Weekday: the day is {'' if value else 'not '}a weekday.",
        "max_speed_variable": f"Maximum speed variability: the maximum speed is {'' if value else 'not '}variable.",
        "unlimited_speed": f"Unlimited speed: the speed is {'not ' if value else ''}limited.",
        "lightning": f"Lightning: there is {'' if value else 'not '}lightning.",
        "incident_reported": f"Incident reports: an incident has {'' if value else 'not '}been reported.",
        "has_junction": f"Junction: this segment is {'' if value else 'not '}part of a junction.",
    }

    categorical_feature_list = [
        "incident_certainty",
        "windGust_speed_word",
        "incident_magnitude",
        "cloudCover_word",
        "cardinal_direction",
        "rain_word",
        "wind_speed_word",
        "landuse",
        "highway",
        "incident_category",
    ]

    if feature in binary_text_map:
        return binary_text_map[feature]
    elif feature in categorical_feature_list:
        if feature == "incident_certainty":
            return f"The certainty of an incident occurring is {convert_probability_to_text(value)}."
        elif feature == "windGust_speed_word":
            return f"Wind gusts: there are {value}."
        elif feature == "incident_magnitude":
            return f"The incident magnitude is {value}."
        elif feature == "cloudCover_word":
            return f"The sky is {value}."
        elif feature == "cardinal_direction":
            return f"The cardinal direction is {value}."
        elif feature == "rain_word":
            return f"Rain: {value}."
        elif feature == "wind_speed_word":
            return f"'Wind: there is {value} wind."
        elif feature == "incident_category":
            return f"The incident category is {convert_incident_to_text(value, article=True)}."
        elif feature == "landuse":
            return f"The area around the road is used for {convert_landuse_to_text(value, article=True)}."
        elif feature == "highway":
            return (
                f"The type of road is {convert_highway_to_text(value, article=False)}."
            )
        raise ValueError(
            f"Feature '{feature}' is categorical but not handled in the text map."
        )
    elif feature == "timestamp_word":
        return f"The timestamp of the segment is {value}."
    elif feature in text_map:
        return text_map[feature].format(value=value)

    raise ValueError(f"Feature '{feature}' not recognized.")


class SingleTokenTask(BaseTask):
    NAME = "single-token-task"
    VERSION = 0
    TASK_LIST = [
        # Category
        "incident_certainty",
        "windGust_speed_word",
        "incident_magnitude",
        "cloudCover_word",
        "cardinal_direction",
        "rain_word",
        "wind_speed_word",
        "landuse",
        "highway",
        "incident_category",
        # Binary
        "oneway",
        "tunnel",
        "bridge",
        "flow_road_closure",
        "weekday",
        "max_speed_variable",
        "unlimited_speed",
        "lightning",
        "incident_reported",
        "has_junction",
        # Numerical
        "lanes",
        "speed_kph",
        "wind_speed",
        "incident_delay",
        "cloudCover",
        "windGust_speed",
        "incident_distance",
        "travel_time",
        "temperature",
        "precipitation",
        "grade_abs",
        "grade",
        "free_flow_speed",
        "current_speed",
        "landuse_distance",
        "length",
        "curvature",
        "bearing",
        "current_travel_time",
        "cos_time",
        "sin_time",
        "delay",
        # Other
        "timestamp_word",
    ]
    NON_TEXT_COLUMNS = []
    MAX_ROUTE_TOKENS = 2900
    SUPPORTS_LLM_POSTPROCESSING = False

    INSERTION_KEY = "<token_sequence::input_route>"

    def get_system_message(
        self, style_info: dict[str, str] | None = None
    ) -> tuple[str, dict[str, str]]:
        raise NotImplementedError("This task does not support LLM post-processing.")

    def get_task(self) -> str:
        """
        Get a random task from the predefined task list.

        Returns:
            str: Task text selected from the task list.

        """
        seleced_feature = random.choice(self.TASK_LIST)
        return seleced_feature

    def generate_task(
        self,
        route_df: pd.DataFrame,
        max_text_length: int = 3 * 29000,
        history: list[dict[str, str]] | None = None,
        style: dict[str, str] | None = None,
        route_path: str | None = None,  # noqa: ARG002
    ) -> list[dict[str, Any]] | None:
        if self.use_llm_for_answering:
            warnings.warn(
                "This task does not support LLM post-processing. "
                "The task will be generated without LLM support."
            )
        cleaned_df = clean_gdf(route_df.copy())

        if cleaned_df is None:
            return None

        # Add necessary columns if they are missing
        cleaned_df["windGust_speed_word"] = (
            cleaned_df["windGust_speed"] - cleaned_df["wind_speed"]
        ).apply(convert_wind_gust_diff_to_word)
        cleaned_df["cardinal_direction"] = cleaned_df["bearing"].apply(
            convert_heading_to_direction
        )
        cleaned_df["cloudCover_word"] = cleaned_df["cloudCover"].apply(
            convert_cloud_cover_to_word
        )
        cleaned_df["wind_speed_word"] = cleaned_df["wind_speed"].apply(
            convert_wind_to_word
        )
        cleaned_df["rain_word"] = cleaned_df["precipitation"].apply(
            convert_precipitation_to_word
        )
        cleaned_df["timestamp_word"] = cleaned_df["timestamp"].strftime("%H:%M:%S")

        task_list = []

        for route_index in range(len(cleaned_df)):
            splice_df = cleaned_df.iloc[[route_index]]

            selected_feature = self.get_task()

            feature_value = splice_df.iloc[route_index][selected_feature]

            message = f"{SingleTokenTask.INSERTION_KEY} {feature_to_text(selected_feature, feature_value)}"

            task_description = {
                "system": None,
                "message": message,
                "style": style,
                "segments": None,
                "header_text": None,
                "address_header_text": None,
                "segment_texts": None,
                "task_text": None,
                "include_header": False,
                "print_step_id": False,
                "use_orig_steps": False,
                "splice_indices": [splice_df["step"].iloc[route_index]],
                "splice_arrays": True,
                "raw_route": None,
                "task_name": self.NAME,
                "task_specific": {
                    "feature": selected_feature,
                    "value": feature_value,
                    "index": route_index,
                    "route_length": len(splice_df),
                },
            }
            task_list.append(self.process_task(task_description, history=history))

        return task_list

    @staticmethod
    def resolve(
        task,
        response,
        tokenization,
        route_formatting_callback,
        add_full_route=True,
        **format_kwargs,
    ):
        if add_full_route:
            response = response.replace(
                SingleTokenTask.INSERTION_KEY,
                route_formatting_callback(tokenization, **format_kwargs),
            )
        else:
            response = response.replace(SingleTokenTask.INSERTION_KEY, "")

        return task["task_text"], response
