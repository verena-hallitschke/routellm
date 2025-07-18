import random

from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask


class SentenceDescriptionTask(BaseTask):
    NAME = "llm-tasks.sentence-description"
    VERSION = 7
    TASK_LIST = [
        "Describe the following route in one sentence:",
        "Hi, could you please describe this route in one sentence?",
        "What would be your description of the following route in one sentence?",
        "I need a very short description of this route (in one sentence):",
        "Can you summarize the following route in a single sentence?",
        "Please provide a brief description of the route in one sentence.",
        "Can you give me a one-sentence overview of the following path?",
        # FEATURE SPECIFIC
        "Describe the weather of the following route in one sentence:",
        "Describe the traffic of the route in one sentence:",
        "Summarize the street characteristics of the route in one sentence:",
    ]

    def get_task(self):
        return random.choice(SentenceDescriptionTask.TASK_LIST)

    def get_system_message(self, style_info=None):
        system_message = "You are an AI that answers questions about routes. Take a deep breath and do it step by step. Make the description as short as possible (one sentence maximum!)."

        if style_info is None:
            style_text, style = self.get_random_style()
        else:
            style = style_info
            style_text = get_style(**style_info)

        return (
            f"{system_message} Follow these phrasing instructions: {style_text}",
            style,
        )
