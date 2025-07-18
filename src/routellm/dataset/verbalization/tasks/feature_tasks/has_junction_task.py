import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class HasJunctionTask(BooleanBaseTask):
    NAME = "feature-tasks.has-junction-task"
    VERSION = 0
    TASK_LIST = [
        "Are there any segments on this route that are part of a junction",
        "Will I encounter any junctions when traveling along this route",
        "Are there any intersections to expect while journeying on this path",
        "Can I anticipate coming across any junctions during my travel on this route",
        "Will my trip along this course involve encountering any crossroads",
        "Are there going to be any meeting points of roads throughout this route's journey",
        "Do I need to be prepared for any road junctures while navigating this itinerary",
        "Is it possible to come across any branching paths when following this specific route",
        "Are there instances of intersecting roads that I may face along this particular path",
        "During my voyage on this route, will I stumble upon any traffic intersections",
        "Can I expect to find any points where roads converge while traversing this course",
        "Will there be situations involving road mergers as I make my way along the chosen route",
        "Hey, I'm travelling on this route and I was wondering whether I will pass any intersections",
        "Any junctions on this route",
        "Will I find intersections here",
        "Route with crossroads",
        "Encountering junctions",
        "Intersections ahead",
        "Junctions during travel",
        "Road convergence points",
        "Mergers on the way",
        "Branching paths ahead",
        "Crossings en route",
        "As I embark on my journey along this particular course, can I anticipate encountering any intersections or points where multiple roads converge",
        "During the entirety of my travel along this designated route, will there be instances where I might come across junctions or crossroads that require navigational attention",
        "When following this specific pathway, would it be necessary for me to prepare for any potential encounters with road junctions, intersections, or other merging points",
        "Throughout the duration of my voyage on this chosen route, should I expect to navigate through any complex road mergers, junctions, or intertwining paths",
        "While undertaking a journey along this path, do you foresee any challenges in the form of crossroads or intersections that could impact my overall traveling experience",
        "As I venture forth on this route's course, can you predict if there will be moments where I'll need to confront and navigate junctions or merging lanes during my travel",
        "In the process of moving through this planned itinerary, are there identifiable instances that involve managing complicated road junctures or points where multiple paths intersect",
        "As I traverse the entirety of this selected route's stretch, should I remain vigilant for any potential occurrences involving traffic crossings or diverging pathways that could influence my travel plans",
        "During my expedition along the specified course laid out before me, is it likely that encountering various junctions and navigating through them will become an integral part of my journey",
        "While steadfastly adhering to the plotted route at hand, can you confirm whether or not circumstances may arise requiring careful navigation around crossroads, intersections, and other multi-directional pathways",
        "So, on this trip, am I gonna bump into any intersections or roads joining together that I should watch out for",
        "While I'm going along this route, will there be any junctions or crossroads that might need some extra attention",
        "Following this path, do you think I'll run into any merging roads or junctions",
        "Following this path, do you think I'll run into any merging roads or junctions that could make things a bit more interesting",
        "As I cruise down this chosen road, should I expect to come across any tricky road mergers, intersections, or winding paths",
        "On this journey, are there any sneaky crossroads or intersections that might put a twist in my overall traveling vibe",
        "Going down this route, can you guess if there'll be times when dealing with junctions or merging lanes becomes part of the adventure",
        "Traveling through this plan, are there moments where I'll have to handle some messy road junctures or spots where paths crisscross",
        "Sticking to the course we've got here, should I keep an eye out for anything like traffic crossings or branching paths that could change things up a bit",
        "On my trek through this course, is it likely that navigating different junctions might become part of the whole experience",
    ]

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, there is an intersection along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are junctions.",
        *[
            f"Yes, you will pass junctions, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are junctions on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, there is no junctions along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are no junctions.",
        *[
            f"No, you will encounter no junctions, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are no junctions on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
        "No, no junctions are encountered.",
    ]
    FEATURE_NAME = "has_junction"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ):
        template = [
            *[
                f"Yes, there {'is a junction' if not plural else 'are multiple junctions'} along the {synonym}.{f' They are located {position}.' if return_position and plural else f' It is located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            f"There {'are junctions' if plural else 'is a junction'}{'' if not return_position else f' {position}'}.",
            *[
                f"Yes, you will pass {'junctions' if plural else 'a junction'}, when traveling along the given {synonym}.{f' They are located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            *[
                f"There {'are junctions' if plural else 'is a junction'} on this {synonym}{f', which are located {position}' if return_position and plural else f', which is located {position}' if return_position else ''}."
                for synonym in ["route", "road", "path", "journey"]
            ],
        ]

        return random.choice(template)
