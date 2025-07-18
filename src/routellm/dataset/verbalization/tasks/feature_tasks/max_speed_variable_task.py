import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class MaxSpeedVariableTask(BooleanBaseTask):
    NAME = "feature-tasks.max-speed-variable-task"
    VERSION = 0
    TASK_LIST = [
        "Are there any roads with a variable speed limit",
        "Are any streets featuring a variable speed limit",
        "Can you find roads that have a changing speed limit",
        "Do any streets incorporate a variable speed limit",
        "Do roads exist with adjustable speed limits",
        "Can you identify routes with a dynamic speed limit in place",
        "Is it possible to locate streets with varying speed limits",
        "Are there any pathways that have a variable maximum allowed speed",
        "Do changeable speed limits occur on certain roadways",
        "Variable speed roads",
        "Changing limit streets",
        "Adjustable speed routes",
        "Dynamic limit highways",
        "Fluctuating limit roads",
        "Varying velocity streets",
        "Variable speed limit",
        "Are there any roads designed with a speed limit that adjusts according to specific conditions",
        "Are there any highways or roadways where the maximum allowed speed varies depending on certain factors",
        "Do any routes feature a dynamic speed limit system, allowing for adjustments based on road conditions",
        "Are there any examples of roads implementing variable speed limits for improved traffic management",
        "Are there any thoroughfares designed with an adaptable speed restriction system for better traffic flow and safety",
        "Are there streets where motorists may encounter fluctuating maximum speeds, depending on circumstances like weather or congestion levels",
    ]

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, there are sections where the maximum speed is variable along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are electric speed signs.",
        *[
            f"Yes, you will encounter areas where the speed limit is determined by an electric speed sign, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are parts of this {synonym} where the maximum allowed speed varies depending on (dynamic) road signs and signals."
            for synonym in ["route", "road", "path", "journey"]
        ],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, the speed limit along the {synonym} is fixed."
            for synonym in ["route", "road", "path"]
        ],
        "There are no roads with a variable speed limit.",
        *[
            f"No, you will encounter no variable speed limits, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"All allowed maximum speeds on this {synonym} are fixed."
            for synonym in ["route", "road", "path", "journey"]
        ],
        "No, no variable maximum speeds are encountered.",
        "No, all streets have fixed speed limits.",
    ]
    FEATURE_NAME = "max_speed_variable"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ):
        template = [
            *[
                f"Yes, there {'is a section with variable speed limit' if not plural else 'are multiple sections with variable speed limits'} along the {synonym}.{f' They are located {position}.' if return_position and plural else f' It is located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            f"There {'are dynamic speed limits' if plural else 'is a street with variable speed limit'}{'' if not return_position else f' {position}'}.",
            *[
                f"Yes, you will encounter {'variable speed limits due to (dynamic) signs or signals' if plural else 'a road with variable allowed maximum speed'}, when traveling along the given {synonym}.{f' They are located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            *[
                f"There {'are streets with variable allowed maximum speed' if plural else 'is a section with variable speed limits'} on this {synonym}{f', which are located {position}' if return_position and plural else f', which is located {position}' if return_position else ''}."
                for synonym in ["route", "road", "path", "journey"]
            ],
        ]

        return random.choice(template)
