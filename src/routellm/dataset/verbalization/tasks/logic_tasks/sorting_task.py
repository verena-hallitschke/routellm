# Bring into ascending/descending order regarding feature X: route A, Route B, Route C


import random

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


class SortingTask(BaseTask):
    # Please give me a subsequence where feature x is < val / == val / > val/ etc.
    NAME = "sorting-task"
    VERSION = 0
    SUPPORTS_LLM_POSTPROCESSING = False
    TASK_LIST = [
        "Bring these routes in {order} regarding the {aggregation} {feature}",
        "Compare the following routes and sort them in {order} based on the {aggregation} {feature}",
        "Given these routes, what would be the correct order regarding the {aggregation} {feature}. Start with the {first_val} value",
        "Please arrange these routes in {order} order considering the {aggregation} {feature}",
        "Sort the listed routes in {order} order with respect to the {aggregation} {feature}",
        "Examine these routes and place them in {order} order according to the {aggregation} {feature}",
        "Rank the mentioned routes from {first_val} value based on their {aggregation} {feature}",
        "Organize these given routes in {order} order, taking into account the {aggregation} {feature}",
        "Could you please put these routes in {order} order based on their {aggregation} {feature}",
        "Can you help me arrange these routes from {first_val} to {opposite_val}, focusing on the {aggregation} {feature}",
        "Sort these routes for me, keeping in mind the {aggregation} {feature}, and go from {first_val} to {opposite_val}",
        "What's the right order for these routes if we're considering the {aggregation} {feature}? Start with the {first_val} one",
        "Let's organize these routes by their {aggregation} {feature}, starting with the route that has the {first_val} value",
    ]

    INSERTION_KEY = "<token_sequence::{name}>"

    def get_task(self):
        if random.random() < 0.5:
            order = "ascending"
            first_val = "lowest"
            opposite_val = "highest"
        else:
            order = "decending"
            first_val = "highest"
            opposite_val = "lowest"

        aggregation = random.choice(["maximum", "minimum", "average", "mean"])

        selected_feature = random.choice(
            list(filter(lambda x: x != "incident_certainty", CONTINUOUS_COLS))
        )

        task_text = (
            super()
            .get_task()
            .format(
                feature=feature_to_text(selected_feature),
                order=order,
                first_val=first_val,
                opposite_val=opposite_val,
                aggregation=aggregation,
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

        return task_text, selected_feature, order, aggregation, use_cot

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

        # if self.use_llm_for_answering:
        #     warnings.warn("This task does not use an LLM interface (use_llm_for_answering is set to True)!")
        #     self.use_llm_for_answering = False

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
            task_text, selected_feature, order, aggregation, use_cot = self.get_task()

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

                for index, name in enumerate(shuffled_split_names):
                    current_split = splice[previous_split:split]
                    answer_splits[shuffled_split_names[index]] = (previous_split, split)
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
                    name = shuffled_split_names[index]
                    current_split = splice[previous_split:split]
                    answer_splits[name] = (previous_split, split)
                    current_split_df = cleaned_df.iloc[
                        current_split
                    ]

                    previous_split = split

                    if aggregation == "maximum":
                        split_results.append(current_split_df[selected_feature].max())
                    elif aggregation == "minimum":
                        split_results.append(current_split_df[selected_feature].min())
                    else:
                        split_results.append(
                            current_split_df[selected_feature].mean()
                        )

                split_results = np.array(split_results).round(5)
                value_df = pd.DataFrame(
                    {"values": split_results, "names": shuffled_split_names}
                )

                value_df = value_df.sort_values(
                    by=["values", "names"], ascending=order == "ascending"
                )

                correct_order = value_df.index.to_numpy()

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
                            aggregation_mode=aggregation,
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

                    value_order_text = ", ".join(
                        [f"{val}{current_unit}" for val in split_results[correct_order]]
                    )

                    split_characters = random.choice(
                        [":", ": ", ":\n", "\n", ":\n\n", "\n\n", " ", "\t"]
                    )

                    cot_prepend_text = random.choice(
                        [
                            f"Let's go through this question step by step. In order to sort the values, we need to know the {aggregation} {feat_text} of each of the routes. These are as follows{split_characters}{value_text}.\nIn the next step, we need to bring the values into the correct order. Since we are searching for the {order} order [{value_order_text}] is the correct order of values.\nUsing this order we can conclude the correct order of routes: ",
                            f"In order to find the correct sequence, we need to calculate the {aggregation} {feat_text} for each of the routes. In the next step we can order the results and order the routes based on the position of their matching value. The {aggregation} {feat_text} for each route is{split_characters}{value_text}.\nAfter bringing these values into the correct order ([{value_order_text}]), we can give the correct answer: ",
                            f"The {aggregation} {feat_text} is{split_characters}{value_text} The values in {order} order are [{value_order_text}].\nTherefore the correct solution is: ",
                            f"We use the {aggregation} {feat_text} of each route to solve this question{split_characters}{value_text} In correct (meaning {order}) order that would be [{value_order_text}].\nThe answer is: ",
                        ]
                    )

                use_bullets = random.random() > 0.5

                correct_order_names = value_df["names"].to_list()

                if use_bullets:
                    bullet_sign = random.choice(["*", "-", "+"])
                    answer_text = (
                        "".join(
                            [
                                f"\t{bullet_sign} {name}\n"
                                for name in correct_order_names
                            ]
                        )
                        + "\n"
                    )
                else:
                    answer_text = (
                        ", ".join(correct_order_names[: num_splits - 1])
                        + f" and{' then' if random.random() < 0.5 else ''} {correct_order_names[num_splits - 1]}."
                    )

                split_characters = random.choice(
                    [":", ": ", ":\n", "\n", ":\n\n", "\n\n", " ", "\t"]
                )

                result_text = random.choice(
                    [
                        f"The correct order when sorting in {order} order regarding the {aggregation} {feat_text} is{split_characters}{answer_text}",
                        f"When sorting the {aggregation} {feat_text} in {order} order {split_characters}{answer_text}",
                        f"The {aggregation} {feat_text} of the given routes in an {order} sequence are{split_characters}{answer_text}",
                        f"Sorted after {aggregation} {feat_text} ({order}){split_characters}{answer_text}",
                        f"{order.capitalize()} order, {aggregation} {feat_text}{split_characters}{answer_text}",
                        f"The correct order is{split_characters}{answer_text}",
                        f"Given the chunks in your request, the correct sequence is{split_characters}{answer_text}",
                        f"This is the correct order{split_characters}{answer_text}",
                        f"Based on the segments you provided, the right order is{split_characters}{answer_text}",
                        f"The proper sequence for these chunks would be{split_characters}{answer_text}",
                        f"Here's how they should be arranged{split_characters}{answer_text}",
                        f"The accurate arrangement of these parts is{split_characters}{answer_text}",
                        f"After sorting them, the true order appears as{split_characters}{answer_text}",
                        f"So, the right order should be{split_characters}{answer_text}",
                        f"Looks like the correct sequence is{split_characters}{answer_text}",
                        f"Here's how it goes{split_characters}{answer_text}",
                        f"After sorting them out, we get this order{split_characters}{answer_text}",
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
                    "aggregation_mode": aggregation,
                    "order": order,
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
            sub_keywords = {
                tag: value[start:end] for tag, value in format_kwargs.items()
            }
            task["task_text"] = task["task_text"].replace(
                SortingTask.INSERTION_KEY.format(name=split_name),
                route_formatting_callback(tokenization[start:end], **sub_keywords),
            )

        return task["task_text"], response
