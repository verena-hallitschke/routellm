import random

from routellm.dataset.routes.const import ICON_CATEGORIES
from routellm.dataset.verbalization.const_to_text import (
    INCIDENT_DESCRIPTIONS,
    convert_incident_to_text,
)
from routellm.dataset.verbalization.tasks.categorical_base_task import (
    CategoricalBaseTask,
)


class IncidentCategoryTask(CategoricalBaseTask):
    NAME = "feature-tasks.incident_category-task"
    VERSION = 0
    FEATURE_NAME = "incident_category"

    FILTER_LIST = ["Unknown", "Cluster"]
    CATEGORIES = ICON_CATEGORIES

    def _get_general_task(
        self,
    ):
        task_list = [
            *[
                f"Are there any incidents on this {word}"
                for word in ["route", "journey", "trip"]
            ],
            "Can you provide an overview of the incidents that might be encountered",
            "Are there any incidents",
            "Are there any traffic incidents during the journey",
            "Are there any reported incidents on this trip",
            "I will be driving the route at hand and I was wondering, if I will encounter any incidents",
        ]

        return random.choice(task_list)

    def _get_specific_value_task(self, target_value):
        text_target = convert_incident_to_text(target_value, article=True)
        plural_target = convert_incident_to_text(target_value, plural=True)
        task_list = [
            f"Are there any {plural_target} along the route",
            f"While passing through this route, are there any instances, where there is {text_target}",
            f"I am driving on this route and I was wondering if there will be {text_target}",
            f"{text_target.capitalize()} on route",
            f"{text_target.capitalize()} while driving",
            f"Any {plural_target} encountered",
            f"Will there be {text_target}",
            f"This will be the journey I am going on today. I just wanted to make sure whether there will be {text_target}",
            f"Is it possible that I find {text_target} when I am following this path",
        ]

        return random.choice(task_list)

    def _get_specific_response_true_single(self, segments, target_value, use_cot):
        text_target = convert_incident_to_text(target_value, article=True)

        if use_cot:
            category_explanation = random.choice(INCIDENT_DESCRIPTIONS[target_value])

            token_reply = random.choice(
                [
                    f"This is indicated by the following tokens: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                    f"The following tokens indicate {text_target}: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                    f"Indicators for {text_target} are: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                    f"These tokens signify {text_target}: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                ]
            )
            reply_list = [
                f"Thank you for your request. {category_explanation} Given the indicators in the route, the entire sequence has {text_target}.",
                f"{category_explanation} Yes, there will be {text_target} during the whole trip.",
                f"{category_explanation} In the given route {text_target} is present.",
                f"{category_explanation} Based on the provided information and the previous description, the answer is yes, there is {text_target} during the route.",
                f"An additional explanation of the meaning and impact of {text_target} aids with answering the question. {category_explanation} When taking the given route into account, yes there is {text_target}.",
                f"{category_explanation}. The answer is yes, there is {text_target}.",
                f"Thank you for your request. As of the provided data, yes, there will be {text_target} for the whole journey. {token_reply}",
                f"Yes, there will be {text_target} during the whole trip. {token_reply}",
                f"In the given route, there is {text_target}. {token_reply}",
                f"Yes, there is {text_target} during the route. {token_reply}",
                f"Indeed, when traveling on the given route {text_target} will be encountered. {token_reply}",
                f"The whole journey takes place, while there is {text_target}. {token_reply}",
            ]
        else:
            reply_list = [
                f"Thank you for your request. As of the provided data, yes, there will be {text_target} for the whole journey.",
                f"Yes, there will be {text_target} during the whole trip.",
                f"In the given route, there is {text_target}.",
                f"Yes, there is {text_target} during the route.",
                f"Indeed, when traveling on the given route {text_target} will be encountered.",
                f"The whole journey takes place, while there is {text_target}.",
            ]

        return random.choice(reply_list)

    def _get_specific_response_true_multi(
        self, segments, target_value, use_cot, position=None, plural=False
    ):
        text_target = convert_incident_to_text(target_value, article=True)
        plural_target = convert_incident_to_text(target_value, plural=True)

        cot_prepend_text = ""

        if use_cot:
            category_explanation = random.choice(INCIDENT_DESCRIPTIONS[target_value])

            cot_prepend_text = random.choice(
                [
                    f"You are asking about a specific type of traffic incident in this route. {category_explanation}\n\nThe correct answer is: ",
                    f"The request aims at understanding the traffic incidents on the provided trip. {category_explanation}\n\nThis brings me to my answer: ",
                    f"To answer this question, it is necessary to understand the different types of incidents that can occur on the road. {category_explanation}\n\nThe answer: ",
                    f"The question can easily be answered with the respective incident definition of the journey. {category_explanation}\n\nAnswer: ",
                    f"An answer to this question can be found by checking traffic incidents that happened along the route. {category_explanation}\n\nThis concludes my answer: ",
                    f"Your question can be answered by inspecting the incident status at several points in the journey. {category_explanation}\n\n",
                    f"You are asking about the incidents on this route. {category_explanation}\n\n",
                    f"The question hints at the traffic incidents on the road. {category_explanation}\n\n",
                    f"The following tokens indicate {text_target}: {self.TOKEN_REASON_REPLACEMENT_TEXT}.\n\n",
                    f"Indicators for {text_target} are: {self.TOKEN_REASON_REPLACEMENT_TEXT}.\n\n",
                    f"These tokens signify {text_target}: {self.TOKEN_REASON_REPLACEMENT_TEXT}.\n\n",
                ]
            )

        if position is not None:
            if plural:
                reply_list = [
                    f"Yes there are multiple instances of {plural_target} {position}.",
                    f"While travelling on this route, there are multiple sections {position} with {plural_target}.",
                    f"In some segments {position} there is {text_target}.",
                    f"Hey, I checked the route you provided and found multiple occasions {position} that match your description.",
                ]
            else:
                reply_list = [
                    f"Yes there is an instance of {text_target} {position}.",
                    f"Hey, I checked the route you provided and found exactly one occasion {position} that matches your description.",
                    f"There is {text_target} {position}.",
                    f"Yes {position} there is {text_target}.",
                ]
        else:
            if plural:
                reply_list = [
                    f"There are multiple occasions with {plural_target}.",
                    f"In more than one section {plural_target} can be encountered.",
                    f"Yes, {plural_target} are present multiple times on this route.",
                    "Hey, I checked the route you provided and found multiple occasions that match your description.",
                    f"Yes at multiple points during the drive {plural_target} can be found.",
                ]
            else:
                reply_list = [
                    f"There is one occasion where {text_target} can be found on the road.",
                    "Yes",
                    f"Given your description I found one passage in the route, with {text_target}.",
                    "Hey, I checked the route you provided and found exactly one occasion that matches your description.",
                    f"Yes there is one part of the route with {text_target}.",
                ]

        return cot_prepend_text + random.choice(reply_list)

    def _get_specific_response_false(self, segments, target_value, use_cot):
        text_target = convert_incident_to_text(target_value, article=True)
        plural_target = convert_incident_to_text(target_value, plural=True)

        cot_prepend_text = ""
        cot_append_text = ""

        if use_cot:
            if random.random() < 0.5:
                category_explanation = random.choice(
                    INCIDENT_DESCRIPTIONS[target_value]
                )

                cot_prepend_text = random.choice(
                    [
                        f"You are asking about a specific type of traffic incident in this route. {category_explanation}\n\nThe correct answer is: ",
                        f"The request aims at understanding the traffic incidents on the provided trip. {category_explanation}\n\nThis brings me to my answer: ",
                        f"To answer this question, it is necessary to understand the different types of incidents that can occur on the road. {category_explanation}\n\nThe answer: ",
                        f"The question can easily be answered with the respective incident definition of the journey. {category_explanation}\n\nAnswer: ",
                        f"An answer to this question can be found by checking traffic incidents that happened along the route. {category_explanation}\n\nThis concludes my answer: ",
                        f"Your question can be answered by inspecting the incident status at several points in the journey. {category_explanation}\n\n",
                        f"You are asking about the incidents on this route. {category_explanation}\n\n",
                        f"The question hints at the traffic incidents on the road. {category_explanation}\n\n",
                    ]
                )
            else:
                cot_append_text = " " + self._categories_to_text(
                    segments[self.FEATURE_NAME].unique().tolist()
                )

        reply_list = [
            f"No, there are no passages in this journey with {plural_target}.",
            f"There will not be {text_target}.",
            f"No, there is not {text_target}.",
            f"No, at no point in the trip will there be {text_target}.",
            f"Hi, I reviewed your request. I have to inform you that given the route in your message there are no sections with {plural_target}.",
            f"No {plural_target} can be found along the route.",
            f"There is no {text_target} or indications for it in the provided route.",
        ]

        return cot_prepend_text + random.choice(reply_list) + cot_append_text

    def _categories_to_text(self, unique_categories, position=None):
        u_cat = [
            cat
            for cat in unique_categories
            if "unknown" not in cat.lower() and "cluster" not in cat.lower()
        ]

        if len(u_cat) <= 1:
            if len(u_cat) == 0:
                if position is None:
                    reply_list = [
                        "There are no traffic incidents.",
                        "During the entirety of the route no incidents have been reported.",
                        "The whole trip is incident free.",
                    ]
                else:
                    reply_list = [
                        f"There are no incidents {position}.",
                        f"{position.capitalize()} no traffic incidents can be found.",
                    ]
            else:
                text_target = convert_incident_to_text(u_cat[0], article=True)
                if position is None:
                    reply_list = [
                        f"There is {text_target} during the whole journey.",
                        f"During the entirety of the route there is {text_target}.",
                    ]
                else:
                    reply_list = [
                        f"There is {text_target} {position}.",
                        f"{position.capitalize()} has {text_target}.",
                    ]

            return random.choice(reply_list)

        incident_text = ""

        use_bullets = random.random() < 0.2

        bullet_sign = random.choice(["*", "-", "+"])

        for val in u_cat:
            if not use_bullets:
                incident_text += f" {convert_incident_to_text(val, plural=True)},"
            else:
                incident_text += (
                    f"\t{bullet_sign} {convert_incident_to_text(val, plural=True)}\n"
                )

        if not use_bullets:
            incident_text = incident_text.strip()[:-1] + "."  # Cut off comma
        else:
            incident_text += "\n"

        split_characters = random.choice(
            [":", ": ", ":\n", "\n", ":\n\n", "\n\n", " ", "\t"]
        )

        if position is None:
            reply_list = [
                f"Here are the types of incidents that are present in this journey{split_characters}{incident_text}",
                f"The following traffic incidents are passed{split_characters}{incident_text}",
                f"On this trip the driver will encounter the following incidents{split_characters}{incident_text}",
            ]
        else:
            reply_list = [
                f"Here are the types of incidents that are present {position}{split_characters}{incident_text}",
                f"{position.capitalize()} the following incidents are passed{split_characters}{incident_text}",
                f"On this part of the journey you will encounter multiple types of incidents. {position.capitalize()} these are precisely{split_characters}{incident_text}",
            ]

        return random.choice(reply_list)
