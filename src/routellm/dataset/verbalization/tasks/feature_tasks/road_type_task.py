import random

from routellm.dataset.routes.const import HIGHWAY
from routellm.dataset.verbalization.const_to_text import convert_highway_to_text
from routellm.dataset.verbalization.tasks.categorical_base_task import (
    CategoricalBaseTask,
)


class RoadTypeTask(CategoricalBaseTask):
    NAME = "feature-tasks.road-type-task"
    VERSION = 0
    FEATURE_NAME = "highway"
    CATEGORIES = HIGHWAY

    def _get_general_task(
        self,
    ):
        task_list = [
            *[
                f"Which types of road are we passing over during this {word}"
                for word in ["route", "journey", "trip"]
            ],
            "Can you provide an overview of the road types",
            "What are the street types on this route",
            "Which types of street are driving on",
            *[
                f"Hey, can you tell me, which types of road I will be driving on when following this {word}"
                for word in ["route", "path", "journey", "road"]
            ],
            "I will be driving the route at hand and I was wondering, which type of streets I will be on",
        ]

        return random.choice(task_list)

    def _get_specific_value_task(self, target_value):
        text_target = convert_highway_to_text(target_value, article=True)
        plural_target = convert_highway_to_text(
            target_value, article=False, plural=True
        )
        task_list = [
            f"Will we be on {text_target}",
            f"While passing through this route, are there any instances, where we are driving on {text_target}",
            f"When on this route, are there any {plural_target}",
            f"{text_target.capitalize()}",
            f"{plural_target} on route",
            f"Any {plural_target} on this trip",
            f"Will there be {text_target}",
            f"This will be the journey I am going on today. I just wanted to make sure whether I'll be on {text_target}",
            f"This is a potential route. Are there any {plural_target} included",
        ]

        return random.choice(task_list)

    def _get_specific_response_true_single(self, segments, target_value, use_cot):
        text_target = convert_highway_to_text(target_value, article=True)
        reply_list = [
            f"Thank you for your request. As of the provided data, yes, the street will be {text_target} for the whole journey.",
            f"Yes, the street type will be {text_target} during the whole trip.",
            f"In the given route, you will be driving over {text_target}.",
            f"Yes, the street is {text_target} during this route.",
            f"Indeed, the road is {text_target}.",
            f"The whole journey takes place on {text_target}.",
        ]
        response = random.choice(reply_list)
        if use_cot:
            response += random.choice(
                [
                    f" This is indicated by the following tokens: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                    f" The following tokens indicate {text_target}: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                    f" Indicators for {text_target} are: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                    f" These tokens signify {text_target}: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                ]
            )

        return response

    def _get_specific_response_true_multi(
        self, segments, target_value, use_cot, position=None, plural=False
    ):
        text_target = convert_highway_to_text(target_value, article=True)
        plural_target = convert_highway_to_text(
            target_value, article=False, plural=True
        )
        if position is not None:
            if plural:
                reply_list = [
                    f"Yes there are multiple instances of the street being {text_target} {position}.",
                    f"While travelling on this route, there are multiple sections {position} where the type of road is {text_target}.",
                    f"There are some segments {position} where you will be driving on {plural_target}.",
                    f"Hey, I checked the route you provided and found multiple occasions {position} that match your description.",
                ]
            else:
                reply_list = [
                    f"Yes there is an instance of the type of road being {text_target} {position}.",
                    f"Hey, I checked the route you provided and found exactly one occasion {position} that matches your description.",
                    f"The road that the driver is on is {text_target} {position}.",
                    f"Yes {position} the road is {text_target}.",
                ]
        else:
            if plural:
                reply_list = [
                    f"There are multiple occasions where the road is {text_target}.",
                    f"In more than one section is the street {text_target}.",
                    f"Yes, you will pass over {plural_target} several times on this route.",
                    "Hey, I checked the route you provided and found multiple occasions that match your description.",
                    f"Yes at multiple points during the drive the type of road is {text_target}.",
                ]
            else:
                reply_list = [
                    f"There is one occasion where the type of road is {text_target}.",
                    "Yes",
                    f"Given your description I found one passage in the route, where the driver is on {text_target}.",
                    "Hey, I checked the route you provided and found exactly one occasion that matches your description.",
                    f"Yes at one part of the route the street is {text_target}.",
                ]
        response = random.choice(reply_list)

        if use_cot:
            response += random.choice(
                [
                    f" This is indicated by the following tokens: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                    f" The following tokens indicate {text_target}: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                    f" Indicators for {text_target} are: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                    f" These tokens signify {text_target}: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                ]
            )

        return response

    def _get_specific_response_false(self, segments, target_value, use_cot):
        text_target = convert_highway_to_text(target_value, article=True)
        plural_target = convert_highway_to_text(
            target_value, article=False, plural=True
        )

        reply_list = [
            f"No, there are no passages in this journey, where the road is {text_target}.",
            f"There are no {plural_target}.",
            f"No, the road is never {text_target} on this route.",
            f"No, at no point in the trip will the road be {text_target}.",
            f"Hi, I reviewed your request. I have to inform you that given the route in your message there are no sections with {plural_target}.",
            f"For the whole route the driver will not be on {plural_target}.",
            f"There are no {plural_target} or indications for them in the provided route.",
        ]

        response = random.choice(reply_list)

        if use_cot:
            response += " " + self._categories_to_text(
                segments[self.FEATURE_NAME].unique().tolist()
            )

        return response

    def _categories_to_text(self, unique_categories, position=None):
        if len(unique_categories) == 1:
            target_text = convert_highway_to_text(unique_categories[0])
            plural_target = convert_highway_to_text(unique_categories[0], plural=True)
            if position is None:
                reply_list = [
                    f"The whole journey is on {plural_target}.",
                    f"During the entirety of the route the road is {target_text}.",
                ]
            else:
                reply_list = [
                    f"The driver is on {target_text} {position}.",
                    f"{position.capitalize()} the road is a {target_text}.",
                ]

            return random.choice(reply_list)

        highway_text = ""

        use_bullets = random.random() < 0.2

        bullet_sign = random.choice(["*", "-", "+"])

        for val in unique_categories:
            if not use_bullets:
                highway_text += f" {convert_highway_to_text(val, plural=True)},"
            else:
                highway_text += (
                    f"\t{bullet_sign} {convert_highway_to_text(val, plural=True)}\n"
                )

        if not use_bullets:
            highway_text = highway_text.strip()[:-1] + "."  # Cut off comma
        else:
            highway_text += "\n"

        split_characters = random.choice(
            [":", ": ", ":\n", "\n", ":\n\n", "\n\n", " ", "\t"]
        )

        if position is None:
            reply_list = [
                f"Here are the types of roads that are present in this journey{split_characters}{highway_text}",
                f"The following roads are passed{split_characters}{highway_text}",
                f"On this trip the driver will traverse the following roads{split_characters}{highway_text}",
            ]
        else:
            reply_list = [
                f"Here are the types of roads that are present {position}{split_characters}{highway_text}",
                f"{position.capitalize()} the following roads are passed{split_characters}{highway_text}",
                f"On this part of the journey you will drive on multiple types of roads. {position.capitalize()} these are precisely{split_characters}{highway_text}",
            ]

        return random.choice(reply_list)
