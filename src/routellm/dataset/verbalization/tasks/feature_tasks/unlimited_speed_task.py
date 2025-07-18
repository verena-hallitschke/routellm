import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class UnlimitedSpeedTask(BooleanBaseTask):
    NAME = "feature-tasks.unlimited-speed-task"
    VERSION = 0
    TASK_LIST = [
        "Are there any streets without a speed limit in this route",
        "Does the journey contain any passages without a speed limit",
        "Are there section with no speed limit",
        "Any streets with no speed limit",
        "Can I find any roads without speed restrictions on this path",
        "Do any parts of the route have unlimited speed zones",
        "Is there a stretch without a speed limit along the way",
        "Are unrestricted speed sections present on this course",
        "In this route, will I encounter areas with no speed limits",
        "Does this path include roadways without designated speed limits",
        "Are there zones with no maximum speeds throughout this journey",
        "Can I expect to find portions with no speed regulations on the way",
        # "On this trip, are there spots where the speed limit is not enforced",
        # "During my travel, will I come across any parts with no speed constraints",
        "No speed limit areas",
        "Limitless speed zones",
        "Any unrestricted roads",
        "Free-speed sections here",
        "Unregulated speed spots",
        "No max-speed stretches",
        "Uncapped speeds ahead",
        "Boundless speed areas",
        "Unlimited speed portions",
        "During the entire length of this journey, are there any specific sections where I won't have to worry about adhering to a particular speed limit",
        "As I navigate through this route, will I come across any roads or highways that don't enforce a maximum speed for drivers to follow",
        "Can you provide information on whether or not there are any stretches along this path where drivers are free to drive without the constraint of a set speed limit",
        "In the course of traversing this route, is it possible that I might encounter certain segments where no legal restrictions on how fast a vehicle can travel exist",
        "Throughout my trip, will there be any locations where road conditions and local regulations permit vehicles to operate without observing a specified maximum speed",
        "While navigating the entirety of this path, can one expect to find any areas in which drivers are not required to adhere to predetermined limits on their traveling speeds",
        "On various segments of this journey, will I pass through zones where motorists can drive without being constrained by established maximum allowable speeds for vehicles on public roads",
        "Is it likely that during my travels along this road system, I may encounter some stretches where traffic laws do not impose specific restrictions regarding vehicular speeds",
        "As part of my trip planning process for this route, am I going to come across sections or streets that do not enforce nor require adherence to any sort of defined vehicular speed limitations",
        "Given the layout and designations along my intended travel course, should I anticipate encountering any segments wherein no formal constraints govern how swiftly motorists may operate their vehicles",
    ]

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, there is a segment with unlimited speed along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are roads without a speed limit.",
        *[
            f"Yes, you will travel on roads without a speed limit, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are roads with unlimited speed on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, there is no roads without speed limit along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "All roads have a speed limit.",
        *[
            f"No, you will encounter only roads with a speed limit, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are no sections without a speed limit on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
        "No, no roads without speed limit are encountered.",
        "No, all roads have limited speed.",
    ]
    FEATURE_NAME = "unlimited_speed"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ):
        template = [
            *[
                f"Yes, there {'is a section without a speed limit' if not plural else 'are multiple passages with no speed limit'} along the {synonym}.{f' They are located {position}.' if return_position and plural else f' It is located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            f"There {'are streets without a speed limit' if plural else 'is a street without speed limit'}{'' if not return_position else f' {position}'}.",
            *[
                f"Yes, you will encounter {'roads with unlimited speed' if plural else 'a road without a speed limit'}, when traveling along the given {synonym}.{f' They are located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            *[
                f"There {'are sections with no speed limit' if plural else 'is a section without a speed limit'} on this {synonym}{f', which are located {position}' if return_position and plural else f', which is located {position}' if return_position else ''}."
                for synonym in ["route", "road", "path", "journey"]
            ],
        ]

        return random.choice(template)
