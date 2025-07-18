import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class FlowRoadClosureTask(BooleanBaseTask):
    NAME = "feature-tasks.flow-road-closure-task"
    VERSION = 0
    TASK_LIST = [
        "Are there any instances where the road is closed",
        "Can you identify any cases of road closures on this path",
        "Are there any occurrences of streets being closed along this journey",
        "Do we have any reports of closed roads throughout this route",
        "Is this route experiencing any road closures at the moment",
        "Have there been any recorded road closures along this course",
        "Are we aware of any road closure incidents on this pathway",
        "Can you confirm if there are any shut down roads along our way",
        "Are we encountering any street closures while travelling through this route",
        "Is it possible that there are some road closures on our current trajectory",
        "Are there indications of closed-off roads during the course of this route",
        "Road closures on route",
        "Closed roads ahead",
        "Any closed streets",
        "Route blockages present",
        "Encountering closures",
        "Pathway obstructions",
        "Shutdown roads nearby",
        "Street closures en route",
        "Blocked route instances",
        "Road closure",
        "Closed road",
        "Are we likely to encounter any road closures along this particular route during our journey",
        "Can you please verify if there have been any reported instances of road closures along the entirety of this route",
        "Is it possible that we might come across any closed roads or other related issues while traveling on this designated path",
        "Have there been any recent notifications or alerts regarding potential road closures along the course of this route",
        "Throughout our journey on this specific route, will we face any instances of temporary or permanent road closures that may impact travel time",
        "In terms of navigating through this planned route, are we to expect any unforeseen complications due to road closures or blockages",
        "Are there any known cases where sections of the intended path have been shut down, causing rerouting or delays along our way",
        "During our travels, is it anticipated that we will stumble upon areas with closed streets or other similar restrictions within the chosen itinerary",
        "As we proceed down the planned trajectory, can you inform us about any instances where access has been limited due to ongoing construction projects or unforeseen incidents leading to road closures",
        "In light of potential obstacles and inconveniences resulting from closed-off roads, are there any such occurrences that we should be prepared for during our journey along this specific route",
        "Within the scope of our current trajectory, can you confirm whether there are any instances where navigational challenges, such as road closures, may present themselves",
    ]

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, there is a road closure along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are closed roads.",
        *[
            f"Yes, you will encounter closed roads, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are road closures on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, there is no closed road along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are no road closures.",
        *[
            f"No, you will encounter no closed roads, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are no road closures on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
        "No, no road closures are encountered.",
        "No, all roads are open.",
    ]
    FEATURE_NAME = "flow_road_closure"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ):
        template = [
            *[
                f"Yes, there {'is a road closure' if not plural else 'are multiple road closures'} along the {synonym}.{f' They are located {position}.' if return_position and plural else f' It is located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            f"There {'are closed roads' if plural else 'is a closed road'}{'' if not return_position else f' {position}'}.",
            *[
                f"Yes, you will encounter {'road closures' if plural else 'a closed road'}, when traveling along the given {synonym}.{f' They are located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            *[
                f"There {'are closed roads' if plural else 'is a closed road'} on this {synonym}{f', which are located {position}' if return_position and plural else f', which is located {position}' if return_position else ''}."
                for synonym in ["route", "road", "path", "journey"]
            ],
        ]

        return random.choice(template)
