import random

import numpy as np

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask
from routellm.dataset.verbalization.text_template import splice_route
from routellm.util.geo_conversion import (
    convert_meters_to_length_scale,
    convert_seconds_to_timescale,
)


class SequenceLocalizationTask(BaseTask):
    NAME = "sequence-localization-task"
    VERSION = 0
    SUPPORTS_LLM_POSTPROCESSING = False
    FULL_INSERTION_KEY = "<full_token_sequence>"
    SELECTION_INSERTION_KEY = "<selection_token_sequence>"
    TASK_LIST = [
        "Is this sequence in the beginning, middle or end of the route based on the {measure}",
        "Can you tell me whether the sequence is in the beginning, middle or end of the route considering the {measure}",
        "Can you find this sequence in the input route? It can either be in the beginning, the end or the middle of it when looking at the {measure}",
        "Could you identify the position of this sequence within the route – is it at the beginning, middle, or end based on the {measure}",
        "Please determine if this sequence appears in the start, middle, or finish of the route with respect to the {measure}",
        "In which part of the route does this sequence occur - is it near the starting point, somewhere in between, or close to the end according to the {measure}",
        "Where can we find this particular sequence along our journey? Is it situated at the onset, halfway through, or towards completion when considering the {measure}",
        "Does this sequence have a placement early on in our path, somewhere midway through our journey, or as we draw closer to our destination based on the {measure}",
        "Can you determine the position of this sequence in the route based on the {measure} - is it at the beginning, middle, or end",
        "Considering the {measure}, where does this sequence appear in the route: at the start, in the middle, or at the finish",
        "In which section of the route can we find this sequence when looking at the {measure} - near the starting point, midway through, or close to the end",
        "Where is this particular sequence located along our journey with respect to the {measure}? At the beginning, halfway through, or near completion",
        "According to the {measure}, is this sequence found early on our path, somewhere in between stops on our journey, or as we approach our destination",
        "Regarding {measure}, can you check if this sequence can be found at either the initial stages of our trip? Or maybe it's somewhere in between important stops? Alternatively, could it be closer to where we want to be",
        "Hey, when considering the {measure}, can you tell me if this sequence shows up at the start, middle, or end of the route",
        "Where do we find this sequence on our journey? Based on the {measure}, is it in the beginning part, somewhere in the middle, or close to wrapping up",
        "Could you check where this series appears along the route? Just want to know if it's near the start, during the middle section, or towards the end according to the {measure}",
        "Can you figure out where this sequence pops up in our trip when considering the {measure}? I'm curious if it's at the very beginning, halfway through, or as we finish up",
    ]

    def get_task(self):
        selected_task = super().get_task()

        measure = random.choice(["travel distance", "current travel time", "distance covered", "travel time including traffic"])

        # Create text for route insertion
        route_insertion_text = random.choice([
            f"Regarding the full route {self.FULL_INSERTION_KEY}\nSequence to find: {self.SELECTION_INSERTION_KEY}.",
            f"The full route {self.FULL_INSERTION_KEY} and this the sequence you have to find: {self.SELECTION_INSERTION_KEY}.",
            f"Full route: {self.FULL_INSERTION_KEY}\nFind: {self.SELECTION_INSERTION_KEY}.",
            f"Find: {self.SELECTION_INSERTION_KEY} in the full route {self.FULL_INSERTION_KEY}",
            f"Given the full route {self.FULL_INSERTION_KEY} find this sequence: {self.SELECTION_INSERTION_KEY}.",
        ])

        use_cot = random.random() < 0.2

        if use_cot:
            selected_task += random.choice([
                "? ", ". ", "! ", "?\n", ".\n", "!\n", "\n", "\n\n"
                ])+ random.choice([
                "Please explain your decision",
                "Provide reasons for your answer",
                "In addition to the answer, explain why it is correct",
                "Explain your answer",
                "Please explain",
            ])

        return selected_task.format(measure=measure) + route_insertion_text, measure, use_cot

    def get_system_message(self, style_info=None):
        system_message = "You are an AI assistant that helps people find information about routes. Keep the language diverse. Make the answer sound how humans would answer to the question. Answer the question, do not describe the route!"

        if style_info is None:
            style_text, style = self.get_random_style()
        else:
            style = style_info
            style_text = get_style(**style_info)

        return f"{system_message} Follow these phrasing instructions: {style_text}", style # f"{system_message} Follow these style instructions: {style_text}", style

    def generate_task(self, route_df, max_text_length: int = 3.0 * 29000, history=None, style=None, route_path=None):
        cleaned_df = clean_gdf(route_df.copy())
        min_seq_length = 50

        if cleaned_df is None:
            return None

        splice_list = [list(range(len(cleaned_df)))]
        if len(cleaned_df) > SequenceLocalizationTask.MAX_ROUTE_TOKENS and self.allow_splicing:
            splice_list = splice_route(len(cleaned_df), SequenceLocalizationTask.MAX_ROUTE_TOKENS)

        task_list = []
        for splice in splice_list:
            splice_df = cleaned_df.iloc[splice]

            system_message, style = self.get_system_message(style_info=style)
            task_text, measure, use_cot = self.get_task()

            use_time = "time" in measure

            if use_time:
                measure_series = cleaned_df["current_travel_time"].cumsum()
                total_measure = cleaned_df["current_travel_time"].sum()
            else:
                measure_series = cleaned_df["length"].cumsum()
                total_measure = cleaned_df["length"].sum()

            # select beginning middle or end
            selected_area = random.choice(["beginning", "middle", "end"])

            first_third = np.where(measure_series < 1/3.0 * total_measure)[0]
            last_third = np.where(measure_series >= 2/3.0 * total_measure)[0]

            first_third_end_index = first_third.max() if len(first_third) > 0 else 0
            last_third_start_index = last_third.min() if len(last_third) > 0 else len(splice_df) - 1

            if selected_area == "beginning":
                last_entry = first_third_end_index
                selected_split = splice[:last_entry + 1]
            elif selected_area == "end":
                first_entry = last_third_start_index
                selected_split = splice[first_entry:]
            else:
                first_entry = first_third_end_index
                last_entry = last_third_start_index
                selected_split = splice[first_entry + 1:last_entry]

            take_random_split = random.random() > 1/3.0

            if take_random_split and len(selected_split) > min_seq_length + 2:

                first_ind = random.randint(0, len(selected_split) - min_seq_length - 1)

                remaining_length = len(selected_split) - first_ind
                split_len = random.randint(min_seq_length, remaining_length)

                selected_split = selected_split[first_ind:first_ind + split_len]

            cot_prepend_text = ""

            if use_cot:
                start_min = min(selected_split)
                start_measure = float(measure_series.iloc[start_min - 1]) if start_min > 0 else 0.0
                end_measure = float(measure_series.iloc[max(selected_split)])

                if use_time:
                    time_scale = random.choice(["s", "h", "m", "detailed"])
                    start_text = convert_seconds_to_timescale(start_measure, time_scale=time_scale, plural=True)
                    end_text = convert_seconds_to_timescale(end_measure, time_scale=time_scale, plural=True)
                    total_text = convert_seconds_to_timescale(total_measure, time_scale=time_scale, plural=True)
                else:
                    length_scale = random.choice(["km", "m", "detailed"])
                    start_text = convert_meters_to_length_scale(start_measure, length_scale=length_scale, plural=True)
                    end_text = convert_meters_to_length_scale(end_measure, length_scale=length_scale, plural=True)
                    total_text = convert_meters_to_length_scale(total_measure, length_scale=length_scale, plural=True)

                cot_prepend_text = random.choice([
                    f"Using the information in the message, I can infer that the segment starts at {start_text}. Furthermore, it ends at {end_text} of the {total_text} route. In order to find the answer to this question, we need an intuition what \"beginning\", \"middle\" and \"end\" mean. Generally the beginning is the first section and the end the last. Assuming the sections have been split up evenly, the beginning would be the first third, the middle the second third and the end the last third.\n\nThe route is split up based on the {measure}, therefore the correct answer is: ",
                    f"Depending on the way the route is split up, beginning, middle and end mean different things. In this case the {measure} is used to split the route into sections. Assuming the route was split evenly, beginning, middle and end are each one third of the total {measure}. The sequence starts at {start_text} and ends at {end_text} of the {measure} (total {total_text}).\nThe answer is: ",
                    f"Generally the beginning is everything that is near the start of the journey. The end of a trip is everything that is close to its destination. Everything else would be specified as the middle. Assuming all parts have the same size, each would be one third of the {measure}.\nIn this case the given sequence starts after {start_text} of the journey and ends after {end_text} of the journey. Since the total journey is {total_text} long, this is clearly located in the {'first' if selected_area == 'beginning' else 'second' if selected_area == 'middle' else 'third'} third of the trip.\nTherefore, the correct answer is: ",
                    f"We assume that all parts, meaning start, middle and end, are of the same size. This results in each spanning over one third of the total {measure}.\n{start_text} is the start {measure} of the sequence. The part we are looking for ends just after {end_text} of the trip. Since the total route is {total_text} long, the sequence is clearly located in the {'first' if selected_area == 'beginning' else 'second' if selected_area == 'middle' else 'third'} third of the trip.\nAnswer: ",
                    f"The provided segment starts at {start_text} and ends at {end_text}. The total {measure} is {total_text}, therefore it is located in the {'1st' if selected_area == 'beginning' else '2nd' if selected_area == 'middle' else '3rd'} third of the route.\nResult: ",
                ])
            message = cot_prepend_text + random.choice([
                f"The sequence is located in the {selected_area} of the journey.",
                f"The selection can be found in the {selected_area} of the provided route.",
                f"When comparing the passage with the full route, it is clear that it is located towards the {selected_area} of it.",
                f"This is part of the {selected_area} of the trip.",
                f"It's part of the {selected_area} of the route.",
                f"We can find the sequence in the {selected_area} of the route."
            ])

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
                    "selected_split": selected_split,
                    "use_cot": use_cot
                }
            }

            task_list.append(self.process_task(task_description, history=history))

        return task_list

    @staticmethod
    def resolve(task, response, tokenization, route_formatting_callback, add_full_route=True, **format_kwargs):

        if add_full_route:
            task["task_text"] = task["task_text"].replace(SequenceLocalizationTask.FULL_INSERTION_KEY, route_formatting_callback(tokenization, **format_kwargs))
        else:
            task["task_text"] = task["task_text"].replace(SequenceLocalizationTask.FULL_INSERTION_KEY, random.choice([
                "in the previous message",
                "in the last messages",
                "from the previous questions"
            ]))

        start_ind = min(task["task_specific"]["selected_split"])
        end_ind = max(task["task_specific"]["selected_split"]) + 1

        sub_kwargs = {
            tag: value[start_ind:end_ind] for tag, value in format_kwargs.items()
        }
        task["task_text"] = task["task_text"].replace(SequenceLocalizationTask.SELECTION_INSERTION_KEY, route_formatting_callback(tokenization[start_ind:end_ind], **sub_kwargs))

        return task["task_text"], response

