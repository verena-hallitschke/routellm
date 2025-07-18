import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class OnewayTask(BooleanBaseTask):
    NAME = "feature-tasks.oneway-task"
    VERSION = 0
    TASK_LIST = [
        "Are there any one-way streets in the route",
        "Does the route include any one-way streets",
        "Are one-way streets present in the route",
        "Can one-way streets be found along the route",
        "Do any one-way streets exist within the route",
        "Is the route made up of any one-way streets",
        "Are there any streets with one-way traffic on this route",
        "Within the route, are there one-way streets",
        "Do we encounter any one-way streets in the designated route",
        "On this particular route, are there streets limited to a single direction of traffic",
        "Does the planned route feature any single-direction streets",
        "Will we encounter any one-way streets",
        "Will we pass through any one-way streets",
        "Are there any one-way streets on this journey",
        "One-way",
        "One-way traffic",
        "One-way streets on route",
        "Route has one-way streets",
        "Any one-ways in route",
        "One-way roads included",
        "One-direction streets present",
        "Route with single-direction roads",
        "Single-way traffic on route",
        "One-way traffic involved",
        "Route features unidirectional streets",
        "Unidirectional roads in route",
        "Are there any streets in the designated route that permit traffic to flow in only one direction",
        "In the given route, can we find streets where vehicles are allowed to travel exclusively in a single direction",
        "Is it possible to encounter streets with one-way traffic regulations within the planned route",
        "Within the specified route, are there any instances of roadways which only permit travel in a singular direction",
        "Does the selected route contain any roads that are strictly designed for one-way vehicular movement",
        "Can we expect to traverse any streets with unidirectional traffic rules along the predetermined route",
        "Are there particular portions of this chosen route that feature roadways with a one-way traffic system in place",
        "As we navigate through the established route, will we come across any streets dedicated solely to single-direction traffic flow",
        "In considering our intended path, is it possible that we'll encounter any street segments where vehicles must adhere to one-way regulations",
        "On this specific journey, are there sections of roadway that dictate a singular direction for vehicle movement",
    ]

    COT_CHANCE = 0.0  # NO COT!!!
    POSITION_CHANCE = 0.0  # NO POSITION

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, there is at least one one-way street along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are one-ways.",
        *[
            f"Yes, you will encounter one-way roads, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are roads with one-way traffic on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, there is no one-way traffic along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are no one-way streets.",
        *[
            f"No, you will encounter no streets with one-way traffic, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are no one-way roads on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
        "No, no one-way streets are encountered.",
        "No, all roads have oncoming traffic.",
    ]
    FEATURE_NAME = "oneway"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ):
        template = [
            *[
                f"Yes, there {'is a one-way street' if not plural else 'are multiple one-way streets'} along the {synonym}.{f' They are located {position}.' if return_position and plural else f' It is located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            f"There {'are multiple street with one-way traffic' if plural else 'is a street with one-way traffic'}{'' if not return_position else f' {position}'}.",
            *[
                f"Yes, you will encounter {'one-way roads' if plural else 'a one-way road'}, when traveling along the given {synonym}.{f' They are located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            *[
                f"There {'are one-way streets' if plural else 'is a one-way street'} on this {synonym}{f', which are located {position}' if return_position and plural else f', which is located {position}' if return_position else ''}."
                for synonym in ["route", "road", "path", "journey"]
            ],
        ]

        return random.choice(template)
