import random

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask
from routellm.dataset.verbalization.text_template import splice_route


class ContinuationTask(BaseTask):
    # Please give me a subsequence where feature x is < val / == val / > val/ etc.
    NAME = "continuation-task"
    VERSION = 0
    SUPPORTS_LLM_POSTPROCESSING = False

    START_INSERTION_KEY = "<token_sequence::sequence_start>"
    ANSWER_INSERTION_KEY = "<token_sequence::{name}>"
    CORRECT_ANSWER_KEY = "<token_sequence::correct_answer>"
    TASK_LIST = [
        "Given the following route, what is the most logical continuation",
        "Which of the following options completes this route",
        "This is a route. How would it continue",
        "How would you complete this route",
        "Please complete the given journey",
        "Given this route, what happens next",
        "Choose the most logical continuation of the given route sequence",
        "Given the following route sequence, select the most appropriate continuation",
        "Select the best next steps in the route sequence",
        "Based on the provided route, pick the most suitable next steps",
        "How would you continue",
        "Identify the most coherent next steps",
        "Find the most fitting continuation of the route described",
        "From the route provided, select the most reasonable next steps",
    ]

    def get_task(self):
        task_text = super().get_task()

        return task_text

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
        num_options = 3
        buffer = 10
        min_option_seq = 10
        max_option_seq = 30
        min_start_sequence = 50
        max_start_sequence = (
            self.MAX_ROUTE_TOKENS
            - num_options * max_option_seq
            - (num_options + 1) * buffer
        )

        if self.use_llm_for_answering:
            raise ValueError(
                "This task does not use an LLM interface (use_llm_for_answering is set to True)!"
            )

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
            task_text = self.get_task()

            len_options = random.randint(min_option_seq, max_option_seq)

            if (
                len(splice) - num_options * len_options - (num_options + 1) * buffer
                <= min_start_sequence
            ):
                task_list.append(None)
                continue

            len_start_sequence = random.randint(
                min_start_sequence,
                min(
                    max_start_sequence,
                    len(splice)
                    - num_options * len_options
                    - (num_options + 1) * buffer,
                ),
            )

            # Pick one option from before the route, one after and one random

            start_sequence_start = random.randint(
                len_options + buffer, len(splice) - len_start_sequence - len_options - 1
            )
            before_split = random.randint(0, start_sequence_start - len_options)
            after_split = start_sequence_start + len_start_sequence

            random_split = random.randint(0, len(splice) - len_options)

            while (
                random_split == before_split
                or after_split <= random_split <= after_split + len_options
            ):
                random_split = random.randint(0, len(splice) - len_options)

            correct_answer_limits = (after_split, after_split + len_options)
            no_correct_answer = False
            if random.random() < 0.2:
                # No answer
                no_correct_answer = True
                new_after_split = random.randint(0, len(splice) - len_options)

                while (
                    new_after_split == before_split
                    or after_split <= new_after_split <= after_split + len_options
                    or random_split == new_after_split
                ):
                    new_after_split = random.randint(0, len(splice) - len_options)

                after_split = new_after_split

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
            )[:num_options]

            task_text += random.choice(
                [
                    f"\n{self.START_INSERTION_KEY}",
                    f"\nSequence: {self.START_INSERTION_KEY}",
                    f"\nReference: {self.START_INSERTION_KEY}",
                    f"\nRoute: {self.START_INSERTION_KEY}",
                ]
            )
            task_extension = " ".join(
                [
                    f"{name}: {self.ANSWER_INSERTION_KEY.format(name=name)}"
                    for name in split_names
                ]
            )

            task_text += "\n" + task_extension.strip()

            random.shuffle(split_names)
            answer_splits = {
                name: (start, start + len_options)
                for name, start in zip(
                    split_names, [after_split, before_split, random_split]
                )
            }

            answer = list(answer_splits.keys())[0]

            if no_correct_answer:
                result_text = random.choice(
                    [
                        "None of the options complete the sequence.",
                        "None of the answers match.",
                        "None of the given options are correct.",
                        "Out of the given options, none would complete this route.",
                        f"None of the answers are correct. The correct continuation is {self.CORRECT_ANSWER_KEY}.",
                    ]
                )
            else:
                result_text = random.choice(
                    [
                        f"The correct answer is {answer}!",
                        f"Option {answer} is the correct continuation.",
                        f"{answer}",
                        f"The logical completion is {answer}.",
                        f"The correct sequence is {answer}!",
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
                "task_name": self.NAME,
                "task_specific": {
                    "limits": answer_splits,
                    "start_sequence_limits": (
                        start_sequence_start,
                        start_sequence_start + len_start_sequence,
                    ),
                    "correct_answer_limits": correct_answer_limits,
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
        sub_kwargs = {
            tag: value[
                task["task_specific"]["start_sequence_limits"][0] : task[
                    "task_specific"
                ]["start_sequence_limits"][1]
            ]
            for tag, value in format_kwargs.items()
        }
        task["task_text"] = task["task_text"].replace(
            ContinuationTask.START_INSERTION_KEY,
            route_formatting_callback(
                tokenization[
                    task["task_specific"]["start_sequence_limits"][0] : task[
                        "task_specific"
                    ]["start_sequence_limits"][1]
                ],
                **sub_kwargs,
            ),
        )
        for split_name, (start, end) in task["task_specific"]["limits"].items():
            sub_kwargs = {tag: value[start:end] for tag, value in format_kwargs.items()}
            task["task_text"] = task["task_text"].replace(
                ContinuationTask.ANSWER_INSERTION_KEY.format(name=split_name),
                route_formatting_callback(tokenization[start:end], **sub_kwargs),
            )

        response = response.replace(
            ContinuationTask.CORRECT_ANSWER_KEY,
            route_formatting_callback(
                tokenization[
                    task["task_specific"]["correct_answer_limits"][0] : task[
                        "task_specific"
                    ]["correct_answer_limits"][1]
                ]
            ),
        )
        return task["task_text"], response
