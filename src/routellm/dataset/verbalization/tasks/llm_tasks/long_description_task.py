import random

from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask


class LongDescriptionTask(BaseTask):
    NAME = "llm-tasks.long-description"
    VERSION = 6
    TASK_LIST = [
        "Give a very long and very detailed description (at least 10 sentences) of the following route:",
        "Hey, can you give me a long description of this route?",
        "Can you provide an extensive and comprehensive account of this route?",
        "Please provide a thorough and in-depth description of the route, covering all the nuances:",
        "I'm interested in a long and detailed description of this route:",
        "Give a complete and detailed summary of this route:",
    ]

    def get_task(self):
        return random.choice(LongDescriptionTask.TASK_LIST)

    def get_system_message(self, style_info=None):
        system_message = "You are an AI that answers questions about routes. Take a deep breath and do it step by step. Use at least 10 sentences!"

        if style_info is None:
            style_text, style = self.get_random_style()
        else:
            style = style_info
            style_text = get_style(**style_info)

        return (
            f"{system_message} Follow these phrasing instructions: {style_text}",
            style,
        )
