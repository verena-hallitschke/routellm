"""Module for the Bridge Task."""

import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class BridgeTask(BooleanBaseTask):
    """Bridge Task for determining the presence of bridges along a route."""

    NAME = "feature-tasks.bridge-task"
    VERSION = 0
    TASK_LIST = [
        "Is there a bridge along this route",
        "Will we be passing over a bridge",
        "Will we encounter a bridge on the road",
        "Is there a bridge along this path",
        "Does this journey take us over a bridge",
        "Is a bridge on the route",
        "Does this route have a bridge",
        "Do we encounter a bridge on this course",
        "A bridge on this path",
        "Bridge along the way",
        "Is passing over a bridge part of the trip",
        "Bridge on route",
        "Is a bridge present here",
        "Any bridges on this journey",
        "Do we cross a bridge",
        "Bridge encountered on route",
        "Route includes a bridge",
        "Will we find a bridge here",
        "Is there a bridge to pass",
        "Bridge on this course",
        "Bridge on this trip",
        "Does the path feature a bridge",
        "Is passing over a bridge part of the route",
        "Does this journey involve crossing a bridge",
        "Can we expect to find a bridge on our way along this path",
        "Will there be an instance of crossing a bridge throughout our journey on this path",
        "Will traversing a bridge occur during our trip",
        "Will traversing a bridge occur during our trip along this route",
        "Bridge",
        "In the course of following this specific route, can we expect to cross a bridge",
        "On this particular route that we are following, will we come across a bridge at any point during our journey",
        "In the course of traversing this chosen path, is there a possibility of encountering a bridge along the way",
        "As we make our way through this specific route, can we expect to find a bridge that we might need to cross or pass by",
        "Bridge nearby",
        "Route's bridge",
        "Bridge ahead",
        "Hi, I'm following this path and I was wondering, whether I will encounter a bridge on it",
        "In accordance with our established travel plan, is there an expectation of finding a bridge at some point during our journey",
        "Crossing any bridges soon",
        "Is there a bridge around here",
        "Got a bridge coming up",
        "Do we pass by a bridge",
        "Are we crossing any bridges",
    ]

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, there is a bridge along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are bridges",
        *[
            f"Yes, you will pass bridges, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are bridges on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, there is no bridge along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are no bridges",
        *[
            f"No, you will pass no bridges, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are no bridges on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
        "No, no bridges are passed.",
    ]
    FEATURE_NAME = "bridge"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ) -> str:
        """
        Generate a response for multi-segment routes based on the presence of bridges.

        Args:
            plural (bool): Indicates if there are multiple bridges.
            return_position (bool): Indicates if the position of the bridges should be \
                included in the response.
            position (str): The position of the bridges, if applicable.

        Returns:
            str: A randomly selected response template indicating the presence of \
                bridges.

        """
        template = [
            *[
                f"Yes, there {'is a bridge' if not plural else 'are multiple bridges'} along the {synonym}.{f' They are located {position}.' if return_position and plural else f' It is located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            f"There {'are bridges' if plural else 'is a bridge'}{'' if not return_position else f' {position}'}.",
            *[
                f"Yes, you will pass {'bridges' if plural else 'a bridge'}, when traveling along the given {synonym}.{f' They are located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            *[
                f"There {'are bridges' if plural else 'is a bridge'} on this {synonym}{f', which are located {position}' if return_position and plural else f', which is located {position}' if return_position else ''}."
                for synonym in ["route", "road", "path", "journey"]
            ],
        ]

        return random.choice(template)
