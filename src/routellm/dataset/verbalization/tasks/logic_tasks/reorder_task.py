import random
import warnings

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask
from routellm.dataset.verbalization.text_template import splice_route


class ReorderTask(BaseTask):
    NAME = "reorder-task"
    VERSION = 0
    SUPPORTS_LLM_POSTPROCESSING = False
    TASK_LIST = [
        "The following segments are all part of one route. Please bring them into the correct order",
        "What is the correct order of the following segments",
        "Please examine these segments, what is the correct order",
        "Assuming all of these chunks originate from the same route, what would be the correct order",
        "These route chunks are in random order. Please put them in the order they appear in the route",
        "I shuffled these segments, please put them back in order",
        "These are all passages of a journey. They are in the wrong sequence. Can you bring them back in order",
        "Please sort these segments in the order they appear in the route",
        "These pieces of a route are mixed up, can you help arrange them in the correct sequence",
        "The order of these route segments is incorrect; please rearrange them to form the proper sequence",
        "Examine these journey fragments and determine their proper arrangement within the route",
        "Please analyze these disordered route segments and establish the correct sequential order",
        "These portions of a trip are out of order; kindly assist in organizing them into their accurate positions",
        "These route parts are mixed up, can you help put them in the right order",
        "The order of these route bits is wrong; please rearrange them to make sense",
        "Look at these journey pieces and figure out how they should be arranged",
        "Please check out these mixed-up route parts and put them in the right order",
        "These bits of a trip are all jumbled up; could you help sort them into the correct order",
    ]

    INSERTION_KEY = "<token_sequence::{name}>"

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
            task_text = self.get_task()

            num_splits = random.randint(3, 26)

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
                "VI",
                "VII",
                "VIII",
                "IX",
                "X",
                "XI",
                "XII",
                "XIII",
                "XIV",
                "XV",
                "XVI",
                "XVII",
                "XVIII",
                "XIX",
                "XX",
                "XXI",
                "XXII",
                "XXIII",
                "XXIV",
                "XXV",
                "XXVI",
            ]

            # Get names for splits
            split_names = random.choice(
                [
                    list("abcdefghijklmnopqrstuvwxyz"),
                    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                    [f"{x}" for x in range(1, 27)],
                    [f"{x})" for x in range(1, 27)],
                    [f"{letter})" for letter in "abcdefghijklmnopqrstuvwxyz"],
                    latin_numerals,
                    [x.lower() for x in latin_numerals],
                    [f"{x.lower()})" for x in latin_numerals],
                ]
            )

            previous_split = 0

            answer_splits = {}

            task_extension = " ".join(
                [
                    f"{split_names[index]}: {self.INSERTION_KEY.format(name=split_names[index])}"
                    for index in range(num_splits)
                ]
            ).strip()

            # Shuffle order
            shuffled_split_names = split_names.copy()[:num_splits]
            random.shuffle(shuffled_split_names)

            for index, split in enumerate(section_split_inds + [len(splice)]):
                answer_splits[shuffled_split_names[index]] = (previous_split, split)
                previous_split = split

            task_text += "\n" + task_extension.strip()

            use_bullets = random.random() > 0.5

            if use_bullets:
                bullet_sign = random.choice(["*", "-", "+"])
                answer_text = (
                    "".join(
                        [
                            f"\t{bullet_sign} {name}\n"
                            for name in shuffled_split_names[:num_splits]
                        ]
                    )
                    + "\n"
                )
            else:
                answer_text = (
                    ", ".join(shuffled_split_names[: num_splits - 1])
                    + f" and{' then' if random.random() < 0.5 else ''} {shuffled_split_names[num_splits - 1]}."
                )

            split_characters = random.choice(
                [":", ": ", ":\n", "\n", ":\n\n", "\n\n", " ", "\t"]
            )

            result_text = random.choice(
                [
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
                    f"The best order for these parts is{split_characters}{answer_text}",
                    f"After sorting them out, we get this order{split_characters}{answer_text}",
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
                    "correct_order": shuffled_split_names,
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
                ReorderTask.INSERTION_KEY.format(name=split_name),
                route_formatting_callback(tokenization[start:end], **sub_kwargs),
            )

        return task["task_text"], response
