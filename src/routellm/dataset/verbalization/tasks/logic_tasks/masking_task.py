import random

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask
from routellm.dataset.verbalization.text_template import splice_route


class MaskingTask(BaseTask):
    # Please give me a subsequence where feature x is < val / == val / > val/ etc.
    NAME = "masking-task"
    VERSION = 0
    SUPPORTS_LLM_POSTPROCESSING = False

    MASK_TOKEN = "<route_mask>"
    ANSWER_KEY = "<token_sequence::answer>"
    TASK_LIST = [
        "Which token is missing in this route",
        "Fill in the missing information",
        "Fill in the missing section",
        "Fill in the part of the route that is missing",
        "Complete this route",
        "Give me the information at the masked part of the route",
        "What is the missing token",
        "What is the missing part of the route",
        f"I marked a part of the route that is missing with {MASK_TOKEN}. Please give me the correct information at that position",
        f"This is a route, but one part is missing. The missing portion is marked as {MASK_TOKEN}, please fill it in",
        f"Complete the missing information marked with '{MASK_TOKEN}'",
        f'Fill in the missing section (marked with "{MASK_TOKEN}")',
        f'Replace "{MASK_TOKEN}" with the correct information',
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
        # mask_length = 1

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

            # Get masking position
            mask_position = random.randint(0, len(splice) - 1)

            result_text = random.choice(
                [
                    f"The correct answer is {MaskingTask.ANSWER_KEY}!",
                    f"The logical completion is {MaskingTask.ANSWER_KEY}.",
                    f"The correct sequence is {MaskingTask.ANSWER_KEY}!",
                    f"The missing information is {MaskingTask.ANSWER_KEY}.",
                    f"The correct replacement for {MaskingTask.MASK_TOKEN} is {MaskingTask.ANSWER_KEY}.",
                    f"The missing section is {MaskingTask.ANSWER_KEY}.",
                    f"Filling in '{MaskingTask.MASK_TOKEN}' with {MaskingTask.ANSWER_KEY} would complete the route.",
                    f"{MaskingTask.ANSWER_KEY} completes the rote.",
                    f"{MaskingTask.ANSWER_KEY}",
                    f'Replacing "{MaskingTask.MASK_TOKEN}" with {MaskingTask.ANSWER_KEY} completes the missing information.',
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
                    "mask_position": mask_position,
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
        before_mask = ""
        if task["task_specific"]["mask_position"] > 0:
            sub_kwargs = {
                tag: value[: task["task_specific"]["mask_position"]]
                for tag, value in format_kwargs.items()
            }
            before_mask = route_formatting_callback(
                tokenization[: task["task_specific"]["mask_position"]], **sub_kwargs
            )

        after_mask = ""
        if task["task_specific"]["mask_position"] < len(tokenization) - 1:
            sub_kwargs = {
                tag: value[task["task_specific"]["mask_position"] + 1 :]
                for tag, value in format_kwargs.items()
            }
            after_mask = route_formatting_callback(
                tokenization[task["task_specific"]["mask_position"] + 1 :], **sub_kwargs
            )

        masked_route = f"{before_mask}{MaskingTask.MASK_TOKEN}{after_mask}"
        task["task_text"] = task["task_text"] + masked_route

        response = response.replace(
            MaskingTask.ANSWER_KEY,
            route_formatting_callback(
                [tokenization[task["task_specific"]["mask_position"]]]
            ),
        )
        return task["task_text"], response
