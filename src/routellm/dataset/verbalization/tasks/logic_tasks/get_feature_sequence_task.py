import random
import warnings

import numpy as np
import pandas as pd

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.routes.const import (
    BOOLEAN_VALUES,
    CAT_FEATURES,
    CONTINUOUS_COLS,
    ENCODER_DICT,
)
from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask
from routellm.dataset.verbalization.text_template import splice_route


def find_longest_true_subset(boolean_feature, route_df):
    true_indices = np.where(boolean_feature)[0]

    if len(true_indices) == 0:
        return []
    subsets = np.split(true_indices, np.where(np.diff(true_indices) != 1)[0] + 1)

    # Find longest True subset
    max_subset = None
    max_length = 0

    for index, subset in enumerate(subsets):
        subset_length = route_df.iloc[subset]["length"].sum()
        if subset_length > max_length:
            max_length = subset_length
            max_subset = subset

    return max_subset.tolist()


class FeatureSequenceTask(BaseTask):
    # Please give me a subsequence where feature x is < val / == val / > val/ etc.
    NAME = "feature-sequence-task"
    VERSION = 0
    SUPPORTS_LLM_POSTPROCESSING = False
    TASK_LIST = [
        "Can you give me a subsequence, where {value_description}",
        "Is there a sequence in this route where {value_description}",
        "Find me a section in this route where {value_description}",
        "Give me a part of the route where {value_description}",
        "Show me the longest possible subsequence where {value_description}",
    ]
    MAX_ROUTE_TOKENS = 750  # Due to subset in response
    INSERTION_KEY = "<token_sequence>"

    def get_system_message(self, style_info=None):
        system_message = "You are an AI assistant that helps people find information about routes. Keep the language diverse. Make the answer sound how humans would answer to the question. Answer the question, do not describe the route!"

        if style_info is None:
            style_text, style = self.get_random_style()
        else:
            style = style_info
            style_text = get_style(**style_info)

        return (
            f"{system_message} Follow these phrasing instructions: {style_text}",
            style,
        )  # f"{system_message} Follow these style instructions: {style_text}", style

    def _generate_bool_task(
        self,
        selected_feature,
        route_df: pd.DataFrame,
    ):
        use_available_target_value = random.choice([True, False])
        mode = random.choice(["eq", "eq", "neq"])

        if use_available_target_value:
            selected_target_value = random.choice([True, False])
        else:
            selected_target_value = random.choice(route_df[selected_feature].unique())

        target_feature = (
            route_df[selected_feature] == selected_target_value
            if mode == "eq"
            else route_df[selected_feature] != selected_target_value
        )

        subset_steps = find_longest_true_subset(target_feature, route_df)

        if mode == "eq":
            text_map = {
                "oneway": "the street is a oneway street",
                "tunnel": "the street is inside a tunnel",
                "bridge": "the road crosses a bridge",
                "flow_road_closure": "the road is blocked",
                "weekday": "the day is a weekday",
                "max_speed_variable": "the speed limit is variable",
                "unlimited_speed": "the maximum speed is unlimited",
                "lightning": "there is a thunderstorm",
                "incident_reported": "a traffic incident has been reported",
                "has_junction": "each segment is part of a junction",
            }
        else:
            text_map = {
                "oneway": "the street is not a oneway street",
                "tunnel": "the street is not inside a tunnel",
                "bridge": "the road does not cross a bridge",
                "flow_road_closure": "the road is not blocked",
                "weekday": "the day is part of the weekend",
                "max_speed_variable": "the speed limit is fixed",
                "unlimited_speed": "the maximum speed is limited",
                "lightning": "there is no thunderstorm",
                "incident_reported": "no traffic incident has been reported",
                "has_junction": "no segment is part of a junction",
            }

        return text_map[selected_feature], subset_steps

    def _generate_continuous_task(
        self,
        selected_feature,
        route_df: pd.DataFrame,
    ):
        mode = random.choice(["eq", "neq", "gt", "gte", "lt", "lte", "int"])

        unique_values = route_df[selected_feature].unique()
        selected_target_value = random.choice(unique_values.tolist())

        remaining_sequence = unique_values[
            unique_values != selected_target_value
        ].tolist()
        if len(remaining_sequence) == 0:
            # nothing left do eq or nq
            mode = random.choice(["eq", "neq"])

        if mode == "eq":
            target_feature = route_df[selected_feature] == selected_target_value
            mode_text = f"is equal to {selected_target_value}"
        elif mode == "neq":
            target_feature = route_df[selected_feature] != selected_target_value
            mode_text = f"is not equal to {selected_target_value}"
        elif mode == "gt":
            target_feature = route_df[selected_feature] > selected_target_value
            mode_text = f"is greater than {selected_target_value}"
        elif mode == "gte":
            target_feature = route_df[selected_feature] >= selected_target_value
            mode_text = f"is greater than or equal to {selected_target_value}"
        elif mode == "lt":
            target_feature = route_df[selected_feature] < selected_target_value
            mode_text = f"is less than {selected_target_value}"
        elif mode == "lte":
            target_feature = route_df[selected_feature] <= selected_target_value
            mode_text = f"is less than or equal to {selected_target_value}"
        elif mode == "int":
            second_target_value = random.choice(remaining_sequence)

            if second_target_value < selected_target_value:
                # Swap
                second_target_value, selected_target_value = (
                    selected_target_value,
                    second_target_value,
                )

            target_feature = (selected_target_value <= route_df[selected_feature]) & (
                route_df[selected_feature] < second_target_value
            )

            mode_text = f"is between {selected_target_value} and {second_target_value}"

        subset_steps = find_longest_true_subset(target_feature, route_df)

        text_map = {
            "lanes": "the number of lanes",
            "length": "the length of each segment",
            "grade": "the grade of the street",
            "speed_kph": "the estimated speed limit",
            "travel_time": "the travel time without traffic",
            "landuse_distance": "the distance to surrounding areas",
            "free_flow_speed": "the free flow speed",
            "curvature": "the curvature",
            "bearing": "each segment's bearing",
            "delay": "the delay",
            "incident_delay": "the delay due to an incident",
            "incident_distance": "the distance to an incident",
            "current_speed": "the current speed",
            "current_travel_time": "the travel time when taking traffic into account",
            "cloudCover": "the cloud coverage",
            "temperature": "the temperature",
            "wind_direction": "the wind direction",
            "wind_speed": "the wind speed",
            "windGust_speed": "the speed of the wind gusts",
            "precipitation": "the precipitation",
            "cos_time": "the cosine of the current time",
            "sin_time": "the sine of the current time",
            "incident_certainty": "the certainty that an incident happened",
            "incident_magnitude": "the magnitude of an incident",
        }

        return f"{text_map[selected_feature]} {mode_text}", subset_steps

    def _generate_cat_task(
        self,
        selected_feature,
        route_df: pd.DataFrame,
    ):
        use_available_target_value = random.choice([True, False])
        mode = random.choice(["eq", "eq", "neq"])

        if use_available_target_value:
            selected_target_value = random.choice(ENCODER_DICT[selected_feature])
        else:
            selected_target_value = random.choice(route_df[selected_feature].unique())

        target_feature = (
            route_df[selected_feature] == selected_target_value
            if mode == "eq"
            else route_df[selected_feature] != selected_target_value
        )

        subset_steps = find_longest_true_subset(target_feature, route_df)

        if mode == "eq":
            text_map = {
                "landuse": {
                    "residential": "there is a residential area nearby",
                    "commercial": "there is a commercial area nearby",
                    "agriculture": "the area around the road is used for agriculture",
                    "institutional": "there are institutions next to the route",
                    "waterbody": "there are bodies of water, like lakes or rivers, around the street",
                    "greenery": "there are green areas near the road (i.e. parks, flowerbeds)",
                    "construction": "there are construction sites on or close to the road",
                    "forest": "there is a forest next to the street",
                    "industrial": "there is an industrial area nearby",
                    "transporation": "the area next to the road is used for transportation",
                    "cemetery": "there is a cemetery nearby",
                    "other": "the area around the road has no significant use",
                },
                "incident_category": {
                    "Unknown": "there are no incidents",
                    "Cluster": "there are no incidents",
                    "Accident": "there is an accident",
                    "Fog": "driving is difficult due to fog",
                    "Dangerous Conditions": "there are dangerous driving conditions",
                    "Rain": "driving is difficult due to heavy rain",
                    "Ice": "driving is difficult due to ice",
                    "Jam": "there is a traffic jam",
                    "Lane Closed": "a lane is closed",
                    "Road Closed": "a road is closed",
                    "Road Works": "there are road works",
                    "Wind": "driving is difficult due to strong wind",
                    "Flooding": "there is flooding",
                    "Broken Down Vehicle": "there is a broken down vehicle",
                },
                "highway": {
                    "unclassified": "the road type is unknown",
                    "secondary": "we are traveling on a secondary road",
                    "tertiary": "the street is a tertiary road",
                    "residential": "we are passing through a residential street",
                    "primary": "we are driving on a primary road",
                    "primary_link": "we are passing a street that links to or from a primary road",
                    "secondary_link": "the road is a link to or from a secondary road",
                    "trunk": "the street type is a trunk road",
                    "living_street": "the driver is on a living street",
                    "trunk_link": "the street links to or from a trunk road",
                    "tertiary_link": "the driver is on a street that links from or to a tertiary road",
                    "motorway_link": "the street is linking to/from a motorway",
                    "motorway": "the route passes a motorway",
                    "busway": "the road is a busway",
                    "disused": "the road is not used anymore",
                    "rest_area": "the driver is in a rest area",
                    "emergency_bay": "the car is in an emergency bay",
                    "road": "the driver is on an unknown type of road",
                },
            }
        else:
            text_map = {
                "landuse": {
                    "residential": "there is no residential area nearby",
                    "commercial": "there is no commercial area nearby",
                    "agriculture": "the area around the road is not used for agriculture",
                    "institutional": "there are no institutions next to the route",
                    "waterbody": "there are no bodies of water, like lakes or rivers, around the street",
                    "greenery": "there are no green areas near the road (i.e. parks, flowerbeds)",
                    "construction": "there are no construction sites on or close to the road",
                    "forest": "there are no forests next to the street",
                    "industrial": "there are no industrial areas nearby",
                    "transporation": "the area next to the road is not used for transportation",
                    "cemetery": "there is no cemetery nearby",
                    "other": "the area around the road has significant use",
                },
                "incident_category": {
                    "Unknown": "there are incidents along the route",
                    "Cluster": "there are incidents along the route",
                    "Accident": "there are no accidents",
                    "Fog": "driving is not difficult due to fog",
                    "Dangerous Conditions": "there are no dangerous driving conditions",
                    "Rain": "driving is not difficult due to heavy rain",
                    "Ice": "driving is not difficult due to ice",
                    "Jam": "there are no traffic jams",
                    "Lane Closed": "no lane is closed",
                    "Road Closed": "the road is not closed",
                    "Road Works": "there are no road works",
                    "Wind": "driving is not difficult due to strong wind",
                    "Flooding": "there is no flooding",
                    "Broken Down Vehicle": "there are no broken down vehicles",
                },
                "highway": {
                    "unclassified": "the road type is known",
                    "secondary": "we are not traveling on a secondary road",
                    "tertiary": "the street is not a tertiary road",
                    "residential": "we are not passing through a residential street",
                    "primary": "we are not driving on a primary road",
                    "primary_link": "we are not passing through a street that links to or from a primary road",
                    "secondary_link": "the road is not a link to or from a secondary road",
                    "trunk": "the street type is not a trunk road",
                    "living_street": "the driver is not on a living street",
                    "trunk_link": "the street does not link to or from a trunk road",
                    "tertiary_link": "the driver is not on a street that links from or to a tertiary road",
                    "motorway_link": "the street is not linking to/from a motorway",
                    "motorway": "the route does not pass a motorway",
                    "busway": "the road is not a busway",
                    "disused": "the road is still used",
                    "rest_area": "the driver is not in a rest area",
                    "emergency_bay": "the car is not in an emergency bay",
                    "road": "the driver is not on an unknown type of road",
                },
            }

        return text_map[selected_feature][selected_target_value], subset_steps

    def generate_task(
        self,
        route_df,
        max_text_length: int = 3.0 * 29000,
        history=None,
        style=None,
        route_path=None,
    ):
        if self.use_llm_for_answering:
            warnings.warn(
                "This task does not use an LLM interface (use_llm_for_answering is set to True)!"
            )
            self.use_llm_for_answering = False

        cleaned_df = clean_gdf(route_df.copy())

        if cleaned_df is None:
            return None

        splice_list = [list(range(len(cleaned_df)))]
        if (
            len(cleaned_df) > FeatureSequenceTask.MAX_ROUTE_TOKENS
            and self.allow_splicing
        ):
            splice_list = splice_route(
                len(cleaned_df), FeatureSequenceTask.MAX_ROUTE_TOKENS
            )

        task_list = []
        for splice in splice_list:
            splice_df = cleaned_df.iloc[splice]
            system_message, style = self.get_system_message(style_info=style)
            task_text = self.get_task()

            selected_feature = random.choice(
                CONTINUOUS_COLS
                + list(filter(lambda x: x != "reversed", BOOLEAN_VALUES))
                + CAT_FEATURES
            )

            if selected_feature in CAT_FEATURES:
                constraint_text, subset_steps = self._generate_cat_task(
                    selected_feature, splice_df
                )
            elif selected_feature in BOOLEAN_VALUES:
                constraint_text, subset_steps = self._generate_bool_task(
                    selected_feature, splice_df
                )
            else:
                constraint_text, subset_steps = self._generate_continuous_task(
                    selected_feature, splice_df
                )

            task_text = task_text.format(value_description=constraint_text)

            if len(subset_steps) == 0:
                result_text = random.choice(
                    [
                        "There are no portions of this route that match your description.",
                        "No subset in this route matches your description.",
                        "Sadly, this route does not match your description.",
                        "Given your description, I could not find a subset that matches the features you described.",
                    ]
                )
            else:
                result_text = random.choice(
                    [
                        f"Sure, the sequence {self.INSERTION_KEY} matches your description.",
                        f"Based on your description {self.INSERTION_KEY} is the longest subsequence.",
                        f"Given the description {self.INSERTION_KEY} is the best match.",
                        f"This part matches your description {self.INSERTION_KEY}.",
                    ]
                )

            message = result_text

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
                "task_name": FeatureSequenceTask.NAME,
                "task_specific": {
                    "target_value": selected_feature,
                    "subset_steps": subset_steps,
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
            task["task_text"] = task["task_text"] + route_formatting_callback(
                tokenization, **format_kwargs
            )

        if FeatureSequenceTask.INSERTION_KEY in response:
            start_ind = min(task["task_specific"]["subset_steps"])
            end_ind = max(task["task_specific"]["subset_steps"]) + 1
            response = response.replace(
                FeatureSequenceTask.INSERTION_KEY,
                route_formatting_callback(tokenization[start_ind:end_ind]),
            )

        return task["task_text"], response
