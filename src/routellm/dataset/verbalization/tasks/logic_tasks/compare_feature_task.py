import random
import warnings

import numpy as np
import pandas as pd

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.routes.const import CONTINUOUS_COLS, UNIT_MAP
from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask
from routellm.dataset.verbalization.text_template import splice_route


def feature_to_text(feature):
    text_map = {
        "lanes": "number of lanes",
        "length": "segment length",
        "grade": "grade of the street",
        "speed_kph": "estimated speed limit",
        "travel_time": "travel time without traffic",
        "landuse_distance": "distance to surrounding areas",
        "free_flow_speed": "free flow speed",
        "curvature": "curvature",
        "bearing": "bearing",
        "delay": "delay",
        "incident_delay": "delay due to an incident",
        "incident_distance": "distance to an incident",
        "current_speed": "current speed",
        "current_travel_time": "travel time when taking traffic into account",
        "cloudCover": "cloud coverage",
        "temperature": "temperature",
        "wind_direction": "wind direction",
        "wind_speed": "wind speed",
        "windGust_speed": "speed of the wind gusts",
        "precipitation": "precipitation",
        "cos_time": "cosine of the current time",
        "sin_time": "sine of the current time",
        "incident_certainty": "certainty that an incident happened",
        "incident_magnitude": "incident magnitude",
    }

    return text_map[feature]


class CompareFeatureTask(BaseTask):
    # Please give me a subsequence where feature x is < val / == val / > val/ etc.
    NAME = "compare-feature-task"
    VERSION = 0
    SUPPORTS_LLM_POSTPROCESSING = False
    TASK_LIST = [
        "Which of these routes has {compare_text}",
        "Can you please state which of these journeys would have {compare_text}",
        "Between these two options, which one has {compare_text}",
        "In terms of routes with {compare_text}, which one should I choose",
        "Considering the criteria of having {compare_text}, which route is preferable",
        "Out of these two routes, can you tell me which has {compare_text}",
        "When it comes to choosing a route with {compare_text}, which option should I pick",
        "Could you help me identify the route that offers {compare_text} between the given choices",
        "For a journey considering {compare_text}, which of these routes is more suitable",
        "Keeping in mind the aspect of possessing {compare_text}, can you recommend one of these routes over the other",
        "Among the multiple available options for these routes, can you help me determine which one fulfills the requirement of having {compare_text}, as this is a key consideration for my journey",
        "In light of several route choices, I'm particularly interested in finding out which one aligns best with the condition of having {compare_text} - could you please assist me in identifying that specific option",
        "As there are multiple routes to choose from, I would greatly appreciate your guidance in pinpointing the ideal choice that satisfies my preference for {compare_text}, so as to make an informed decision on my journey.",
    ]

    INSERTION_KEY = "<token_sequence::{name}>"

    def get_task(self):
        aggregation_mode = random.choice(["maximum", "minimum", "average", "mean"])
        compare_mode = random.choice(["higher", "lower"])

        selected_feature = random.choice(
            list(filter(lambda x: x != "incident_certainty", CONTINUOUS_COLS))
        )

        task_text = (
            super()
            .get_task()
            .format(
                compare_text=f"the {compare_mode} {aggregation_mode} {feature_to_text(selected_feature)}"
            )
        )

        use_cot = random.random() < 0.2

        if use_cot:
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

        return task_text, selected_feature, compare_mode, aggregation_mode, use_cot

    def get_system_message(self, style_info=None):
        system_message = "You are an AI assistant that helps people find information about routes. Keep the language diverse. Make the answer sound how humans would answer to the question. Answer the question, do not describe the route! Give short and precise answers. The routes that should be compared are formatted the following way:\n<name_route_1>: <description_route_1>\n\n----\n<name_route_2>: <description_route_2>\n\n----"

        if style_info is None:
            style_text, style = self.get_random_style()
        else:
            style = style_info
            style_text = get_style(**style_info)

        return (
            f"{system_message} Follow these phrasing instructions: {style_text}",
            style,
        )  # f"{system_message} Follow these style instructions: {style_text}", style

    def generate_task(
        self,
        route_df,
        max_text_length: int = 3.0 * 29000,
        history=None,
        style=None,
        route_path=None,
    ):
        min_seq = 20

        if self.use_llm_for_answering:
            warnings.warn(
                "This task does not use an LLM interface (use_llm_for_answering is set to True)!"
            )
            self.use_llm_for_answering = False

        cleaned_df = clean_gdf(route_df.copy(), double_digit_grade=True)

        if cleaned_df is None:
            return None

        splice_list = [list(range(len(cleaned_df)))]
        if len(cleaned_df) > self.MAX_ROUTE_TOKENS and self.allow_splicing:
            splice_list = splice_route(len(cleaned_df), self.MAX_ROUTE_TOKENS)

        task_list = []
        for splice in splice_list:
            splice_df = cleaned_df.iloc[splice]
            system_message, style = self.get_system_message(style_info=style)
            task_text, selected_feature, compare_mode, aggregation_mode, use_cot = (
                self.get_task()
            )

            num_splits = random.randint(2, 5)

            section_split_inds = []

            remaining_numbers = list(range(min_seq, len(splice) - min_seq))

            for _ in range(num_splits - 1):
                if len(remaining_numbers) == 0:
                    # early stop
                    num_splits = len(section_split_inds) + 1
                    break
                # make sure that the sequences are at least min_seq long
                random_split = random.choice(remaining_numbers)

                section_split_inds.append(random_split)

                for num in range(
                    max(0, random_split - min_seq), random_split + min_seq
                ):
                    try:
                        # delete
                        remaining_numbers.remove(num)
                    except ValueError:
                        # Number not present
                        continue

            section_split_inds = sorted(section_split_inds)

            latin_numerals = [
                "I",
                "II",
                "III",
                "IV",
                "V",
            ]

            # Get names for splits
            split_names = random.choice(
                [
                    list("abcde"),
                    list("ABCDE"),
                    [f"{x}" for x in range(1, 6)],
                    [f"{letter})" for letter in "abcde"],
                    latin_numerals,
                    [x.lower() for x in latin_numerals],
                    [f"{x.lower()})" for x in latin_numerals],
                    [f"{x})" for x in range(1, 6)],
                ]
            )

            previous_split = 0
            split_results = []

            answer_splits = {}

            task_extension = " ".join(
                [
                    f"{name}: {self.INSERTION_KEY.format(name=name)}"
                    for name in split_names[:num_splits]
                ]
            )

            task_text += "\n" + task_extension.strip()

            shuffled_split_names = split_names[:num_splits].copy()
            random.shuffle(shuffled_split_names)

            if self.use_llm_for_answering:
                message = f"{task_text}"

                for index, split in enumerate(section_split_inds + [len(splice)]):
                    name = shuffled_split_names[index]
                    current_split = splice[previous_split:split]
                    answer_splits[name] = (previous_split, split)
                    current_split_df = cleaned_df.iloc[
                        current_split
                    ]

                    previous_split = split

                    textual_description = self.convert_to_text(current_split_df)

                    message = message.replace(
                        f"{self.INSERTION_KEY.format(name=name)}",
                        f"{textual_description}\n\n----\n",
                    )

            else:
                for index, split in enumerate(section_split_inds + [len(splice)]):
                    current_split = splice[previous_split:split]
                    answer_splits[shuffled_split_names[index]] = (previous_split, split)
                    current_split_df = cleaned_df.iloc[
                        current_split
                    ]

                    previous_split = split

                    if aggregation_mode == "maximum":
                        split_results.append(current_split_df[selected_feature].max())
                    elif aggregation_mode == "minimum":
                        split_results.append(current_split_df[selected_feature].min())
                    else:
                        split_results.append(
                            current_split_df[selected_feature].mean()
                        )

                split_results = np.array(split_results).round(5)
                value_df = pd.DataFrame(
                    {"values": split_results, "names": shuffled_split_names}
                )

                value_df = value_df.sort_values(by=["values", "names"], ascending=True)

                if compare_mode == "lower":
                    answer_value = value_df["values"].min()
                else:
                    answer_value = value_df["values"].max()

                answer = value_df[value_df["values"] == answer_value]["names"].to_list()

                cot_prepend_text = ""
                feat_text = feature_to_text(selected_feature)

                current_unit = UNIT_MAP.get(selected_feature)
                if current_unit is None:
                    current_unit = ""
                else:
                    current_unit = f" {current_unit}"

                if use_cot:
                    use_bullets = random.random() > 0.5

                    if use_bullets:
                        text_template = random.choice(
                            [
                                "\t{bullet_sign} {split_name}: {aggregation_mode} of {val}{current_unit}\n",
                                "\t{bullet_sign} the {aggregation_mode} of {split_name} is {val}{current_unit}\n",
                            ]
                        )
                    else:
                        text_template = random.choice(
                            [
                                " {split_name} has a {aggregation_mode} value of {val}{current_unit}.",
                                " {split_name} - {aggregation_mode} value: {val}{current_unit}.",
                            ]
                        )

                    bullet_sign = random.choice(["*", "-", "+"])

                    format_dict = {}
                    for index, val in enumerate(split_results):
                        format_dict[shuffled_split_names[index]] = text_template.format(
                            split_name=shuffled_split_names[index],
                            aggregation_mode=aggregation_mode,
                            val=val,
                            bullet_sign=bullet_sign,
                            current_unit=current_unit,
                        )

                    value_text = "".join(
                        {
                            letter: format_dict[letter]
                            for letter in split_names[:num_splits]
                        }.values()
                    )
                    if not use_bullets:
                        value_text = value_text.strip()
                    else:
                        value_text += "\n"

                    split_characters = random.choice(
                        [":", ": ", ":\n", "\n", ":\n\n", "\n\n", " ", "\t"]
                    )

                    cot_prepend_text = random.choice(
                        [
                            f"The question about the {compare_mode} {aggregation_mode} {feat_text} can be easily answered by comparing the feature values of the segments{split_characters}{value_text} ",
                            f"In order to find the {compare_mode} {aggregation_mode} {feat_text} we check the feature values{split_characters}{value_text} Therefore: ",
                            f"The feature values give us indicators to answer the question about the {compare_mode} {aggregation_mode} {feat_text}{split_characters}{value_text} The correct answer is: ",
                            f"Based on the feature information in the sequence we can make the following assertions{split_characters}{value_text} We deduce the correct answer: ",
                        ]
                    )

                if len(answer) == 1:
                    answer_name = answer[0]
                    result_text = random.choice(
                        [
                            f"{answer_name} has the {compare_mode} value!",
                            f"Route {answer_name} has the {compare_mode} {aggregation_mode} {feat_text}.",
                            f"{answer_name}",
                            f"It is route {answer_name}!",
                            f"The route with the {compare_mode} {aggregation_mode} {feat_text} is {answer_name}.",
                        ]
                    )
                else:
                    answer_name = (
                        ", ".join(answer[: len(answer) - 1]) + f" and {answer[-1]}"
                    )

                    result_text = random.choice(
                        [
                            f"Since these routes have the same value, {answer_name} have the {compare_mode} values!",
                            f"Routes {answer_name} have the {compare_mode} {aggregation_mode} {feat_text}.",
                            f"{answer_name}",
                            f"It is route {answer_name}!",
                            f"The routes with the {compare_mode} {aggregation_mode} {feat_text} are {answer_name}. This is because all {len(answer)} have the same value.",
                        ]
                    )

                message = cot_prepend_text + result_text

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
                    "limits": answer_splits,
                    "target_value": selected_feature,
                    "aggregation_mode": aggregation_mode,
                    "compare_mode": compare_mode,
                    "use_cot": use_cot,
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
        for split_name, (start, end) in task["task_specific"]["limits"].items():
            sub_kwargs = {tag: value[start:end] for tag, value in format_kwargs.items()}
            task["task_text"] = task["task_text"].replace(
                CompareFeatureTask.INSERTION_KEY.format(name=split_name),
                route_formatting_callback(tokenization[start:end], **sub_kwargs),
            )

        return task["task_text"], response
