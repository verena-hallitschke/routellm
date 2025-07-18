import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class TunnelTask(BooleanBaseTask):
    NAME = "feature-tasks.tunnel-task"
    VERSION = 0
    TASK_LIST = [
        "Is there a tunnel along this route",
        "Will we be passing through a tunnel",
        "Will we encounter a tunnel on the road",
        "Is there a tunnel along this path",
        "Does this journey take us through tunnels",
        "Are there any tunnels on this route we're taking",
        "Can we expect to go through a tunnel during our trip",
        "Do we come across any tunnels while traveling this way",
        "Does this route have a tunnel that we'll pass through",
        "Is a tunnel part of the path we are going to take",
        "Will our journey involve going through any tunnels",
        "Are we likely to find a tunnel on this particular route",
        "Can I anticipate encountering a tunnel along our way",
        "During our travel, will there be a tunnel to pass through",
        "As we follow this course, is there a tunnel in store for us",
        "Tunnel on route",
        "Passing a tunnel",
        "Tunnel ahead",
        "Tunnel nearby",
        "Any tunnels here",
        "Route with tunnel",
        "Tunnel on path",
        "Going through tunnel",
        "Tunnel encounter",
        "Journey with tunnel",
        "As we embark on this journey, can we anticipate navigating through any tunnels along the way",
        "During our travels, will there be any instances where we encounter a tunnel as part of our route",
        "Is it possible that we'll come across an underground passage or tunnel while traversing this particular path",
        "Throughout the entirety of our journey, can we expect to pass through or alongside any tunnels that may be present on our route",
        "As we make our way along this specific course, is there a likelihood that we'll find ourselves traveling through a tunnel at some point",
        "Considering the route we have chosen for our trip, will there be any opportunities to experience passing through a tunnel during our travels",
        "While taking this particular path during our journey, is it plausible that we might come across or travel through one or more tunnels along the way",
        "Given the nature of our intended route and its features, is it reasonable to assume that we might encounter a tunnel during some portion of our journey",
        "Do the characteristics of this travel path suggest that at any point in time during our journey, there could be a possibility of encountering a tunnel to navigate through",
        "In light of the specific route chosen for this trip, should we prepare ourselves for potentially encountering and traveling through a tunnel along the way",
    ]

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, there is a tunnel along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are tunnels.",
        *[
            f"Yes, you will encounter tunnels, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are tunnels on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, there is no tunnel along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are no tunnels.",
        *[
            f"No, you will encounter no tunnels, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are no tunnels on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
        "No, no tunnels are encountered.",
    ]
    FEATURE_NAME = "tunnel"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ):
        template = [
            *[
                f"Yes, there {'is a tunnel' if not plural else 'are multiple tunnels'} along the {synonym}.{f' They are located {position}.' if return_position and plural else f' It is located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            f"There {'are tunnels' if plural else 'is a tunnel'}{'' if not return_position else f' {position}'}.",
            *[
                f"Yes, you will encounter {'tunnels' if plural else 'a tunnel'}, when traveling along the given {synonym}.{f' They are located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            *[
                f"There {'are tunnels' if plural else 'is a tunnel'} on this {synonym}{f', which are located {position}' if return_position and plural else f', which is located {position}' if return_position else ''}."
                for synonym in ["route", "road", "path", "journey"]
            ],
        ]

        return random.choice(template)
