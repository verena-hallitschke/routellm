import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class LightningTask(BooleanBaseTask):
    NAME = "feature-tasks.lightning-task"
    VERSION = 0
    TASK_LIST = [
        "Have there been any reports of lightning on the path",
        "Have there been any reports of thunderstorms on the path",
        "Can you find any instances of lightning or storms along the journey",
        "Are there records of thunder storms occurring on this route",
        "Do we have evidence of lightning strikes throughout the way",
        "Are there any documented cases of lightning events along this road",
        "Has anyone reported encountering lightning during their travels on this route",
        "Is there mention of thunderous downpours while traversing the path",
        "Were there occurrences of lightning strikes on that course",
        "Lightning on route",
        "Thunderstorms along way",
        "Storms in path reports",
        "Lightning",
        "Thunder",
        "Lightning, storm documents",
        "Thunderstorm",
        "Are there any documented instances where individuals traveling along this particular route encountered either flashing bolts of lightning or intense, boisterous thunderstorms during their journey",
        "In an extensive search for information regarding meteorological phenomena occurring within proximity to our chosen travel course, are we able to gather sufficient evidence showcasing sightings and experiences with lightning and thunder",
        "Can it be determined whether individuals traversing the specified route have ever encountered natural phenomena such as lightning strikes or thunderstorms, based on reports or documented evidence available",
    ]

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, there is lightning along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are thunder storms.",
        *[
            f"Yes, you will encounter lightning, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are occurrences of thunder and lightning on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, there is no lightning along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are no thunder storms.",
        *[
            f"No, you will not encounter lightning, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are no thunder storms on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
        "No, no lightning is encountered.",
    ]
    FEATURE_NAME = "lightning"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ):
        template = [
            *[
                f"Yes, there {'is lightning' if not plural else 'are multiple accounts of lightning'} along the {synonym}.{f' They are located {position}.' if return_position and plural else f' It is located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            f"There {'are storms' if plural else 'is lightning'}{'' if not return_position else f' {position}'}.",
            *[
                f"Yes, you will encounter {'multiple thunderstorms' if plural else 'lightning'}, when traveling along the given {synonym}.{f' They are located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            *[
                f"There {'are multiple thunder storms' if plural else 'is lightning'} on this {synonym}{f', which are located {position}' if return_position and plural else f', which is located {position}' if return_position else ''}."
                for synonym in ["route", "road", "path", "journey"]
            ],
        ]

        return random.choice(template)
