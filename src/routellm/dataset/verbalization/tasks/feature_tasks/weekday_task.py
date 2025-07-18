import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class WeekdayTask(BooleanBaseTask):
    NAME = "feature-tasks.weekday-task"
    VERSION = 0
    TASK_LIST = [
        "Is this trip during the week",
        "Was this recorded on a weekday",
        "Did this journey take place on a weekday",
        "Did I drive this route during the week",
        "Is this route taken on weekdays",
        "Was the route followed on a weekday",
        "Did I travel on this route during a week day",
        "Is this a route commonly used on weekdays",
        "Did the journey occur during the week",
        "Was this specific route driven on a weekday",
        "Is this path taken during the workweek",
        "Was it a weekday when the provided route was recorded",
        "Did we follow this route during the working days",
        "Has this route been driven on a typical weekday",
        "Weekday route",
        "Driven midweek",
        "Route on workdays",
        "Weekday trip taken",
        "Took path in-week",
        "Workweek journey",
        "Weekday drive done",
        "Traveled weekdays",
        "Drove during week",
        "Midweek navigation",
        "Can you confirm if this particular driving route was undertaken specifically during one of the weekdays",
        "I'm trying to recall if my decision to embark upon this particular course occurred on a weekday – can you provide any insight",
        "Is it fair to assume that my usual driving path was taken on a weekday",
        "In reflecting upon the likely scenarios surrounding our traversal along said path, is there evidence pointing towards its occurrence within the work week",
        "Would it be safe to conclude that our chosen adventure down this roadway was contained within weekday confines",
        "Could it be possible that our voyage upon said street was experienced amid what many might consider an ordinary working day",
        "Might there be reason enough for us to deduce that said trip transpired amidst weekdays",
    ]
    COT_CHANCE = 0.0  # NO COT
    POSITION_CHANCE = 0.0  # NO POSITION

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, the drive along the {synonym} was during a weekday."
            for synonym in ["route", "road", "path"]
        ],
        "The trip was during a weekday.",
        *[
            f"Yes, traveling along the given {synonym} happened during the week."
            for synonym in ["route", "road", "path"]
        ],
        # *[f"The {synonym} was not captured during the weekend." for synonym in ["route", "road", "path", "journey"]],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, the trip along the given {synonym} was on the weekend."
            for synonym in ["route", "road", "path"]
        ],
        "No it's during the weekend.",
        *[
            f"No, the day when traveling along the given {synonym} was not a weekday."
            for synonym in ["route", "road", "path"]
        ],
        # *[f"There are no road closures on this {synonym}." for synonym in ["route", "road", "path", "journey"]],
        "No, the trip is on the weekend",
        "No, it's not during the week.",
    ]
    FEATURE_NAME = "weekday"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ):
        route_word = random.choice(["trip", "route", "path", "journey", "voyage"])

        if not return_position:
            return f"Yes the {route_word} is partly on a weekday."

        if "end" in position:
            return f"Yes the {route_word} starts on the weekend and ends on a weekday."

        elif "beginning" in position:
            return (
                f"Yes the {route_word} starts during the week and ends on the weekend."
            )

        elif "middle" in position:
            return f"Yes the {route_word} starts on the weekend continues through the week and ends on the weekend."

        return f"Multiple segments of thus {route_word} are during the week. They are spread {position}."
