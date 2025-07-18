import random

from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask


class ShortDescriptionTask(BaseTask):
    NAME = "llm-tasks.short-description"
    VERSION = 6
    TASK_LIST = [
        "Briefly describe the following route in up to 5 sentences:",
        "Give a brief description of the following route:",
        "Please summarize the route in a few sentences.",
        "Can you provide a concise overview of this route?",
        "Can you please briefly describe this route?",
        "I'm interested in a brief description of this route:",
    ]

    def get_task(self):
        return random.choice(ShortDescriptionTask.TASK_LIST)

    def get_system_message(self, style_info=None):
        system_message = "You are an AI that answers questions about routes. Take a deep breath and do it step by step. Use a maximum of 5 sentences."

        if style_info is None:
            style_text, style = self.get_random_style()
        else:
            style = style_info
            style_text = get_style(**style_info)

        return (
            f"{system_message} Follow these phrasing instructions: {style_text}",
            style,
        )
