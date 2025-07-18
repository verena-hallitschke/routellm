import random

import numpy as np

from routellm.dataset.routes.const import (
    PRECIPITATION_LIMITS,
    convert_precipitation_to_word,
)
from routellm.dataset.verbalization.tasks.categorical_base_task import (
    CategoricalBaseTask,
)


class PrecipitationTask(CategoricalBaseTask):
    NAME = "feature-tasks.precipitation-task"
    VERSION = 0
    FEATURE_NAME = "precipitation"
    CATEGORIES = [*PRECIPITATION_LIMITS.keys()]

    def _get_general_task(
        self,
    ):
        task_list = [
            *[
                f"How high is the precipitation during this {word}"
                for word in ["route", "journey", "trip"]
            ],
            "Can you provide an overview of the precipitation",
            "I will be driving the route at hand and I was wondering, if it will be raining",
            "What is the precipitation level along this path",
            "Can you tell me the amount of rainfall on this route",
            "What's the extent of precipitation on this journey",
            "How much precipitation can be expected on this route",
            "What are the precipitation levels for this particular route",
            "How significant is the rainfall on this path",
            "What's the rainfall situation on this route",
            "How much rain are we likely to encounter on this route",
            "What is the intensity of precipitation along this journey",
            "What's the rain forecast for this route",
            "Can you tell me the rain outlook for this journey",
            "How's the rain forecast looking for this itinerary",
            "What are the rain predictions for this course",
            "Is there any rain anticipated on this route",
            "What's the weather forecast regarding rain along this way",
            "Can you provide the rain prognosis for this route",
            "Are there any rain showers expected on this path",
            "What's the rain situation for this specific route",
        ]

        return random.choice(task_list)

    def _get_specific_value_task(self, target_value):
        task_list = [
            f"Is {target_value} expected along the route",
            f"Can we anticipate {target_value} on our journey",
            f"Will we encounter {target_value} during our travel",
            f"Is it likely to have {target_value} on the way",
            f"Should we prepare for {target_value} while traveling",
            f"Are there chances of {target_value} on the route",
            f"Is the route going to have {target_value}",
            f"Will the route be affected by {target_value}",
            f"Can we predict {target_value} on the path",
            f"Is the presence of {target_value} expected on the route",
            f"Considering the precipitation conditions, is it possible that we might experience {target_value} as we make our way along the route",
            f"Taking into account the potential weather changes, should we be prepared to face {target_value} precipitation while traversing the route",
            f"Given the varying nature of precipitation, is there a likelihood of encountering {target_value} during our journey on the route",
            f"Hey, do you think we might run into {target_value} on our way today",
            f"Just curious, will we have to deal with {target_value} during our trip",
            f"I was wondering, is {target_value} something we might come across on the route",
            f"So, is {target_value} going to be a part of our journey today",
            f"Are we expecting to face {target_value} while we're out and about today",
        ]

        return random.choice(task_list)

    def _get_specific_response_true_single(self, segments, target_value, use_cot):
        cot_pretext = ""
        if use_cot:
            # get value range
            min_val = round(segments["precipitation_value"].min())
            max_val = round(segments["precipitation_value"].max())

            if min_val == max_val:
                range_text = random.choice(
                    [
                        f"constantly {min_val} dBZ",
                        f"constantly {min_val} dBZ",
                        f"{min_val} dBZ",
                        f"persistently {min_val} dBZ",
                    ]
                )
            else:
                range_text = random.choice(
                    [
                        f"in the range from {min_val} to {max_val} dBZ",
                        f"between {min_val} and {max_val} dBZ",
                        f"in the interval [{min_val}, {max_val}] dBZ",
                    ]
                )

            cot_pretext = random.choice(
                [
                    f"Based on the available data, the precipitation levels measured in dBZ are {range_text}. This indicates that the correct answer is: ",
                    f"Considering the precipitation measurements in dBZ, which are {range_text}, it is clear that: ",
                    f"Taking into account the measured precipitation levels in dBZ, specifically {range_text}, we can conclude that: ",
                    f"The precipitation data, which shows dBZ values {range_text}, leads us to the following conclusion: ",
                    f"With the precipitation measurements in dBZ being {range_text}, it becomes evident that: ",
                    f"Upon analyzing the precipitation levels in dBZ, which are {range_text}, we can confidently state that: ",
                    f"Given the recorded precipitation measurements in dBZ, falling within {range_text}, we can determine that: ",
                    f"After examining the precipitation data with dBZ values {range_text}, we can affirm that: ",
                    f"By evaluating the precipitation measurements in dBZ, which are {range_text}, it is apparent that: ",
                    f"Through the assessment of the precipitation levels in dBZ, specifically {range_text}, we can deduce that: ",
                    f"Based on the available data, the precipitation levels measured in dBZ are {range_text}. This indicates that the correct answer is: ",
                    f"Considering the precipitation measurements in dBZ, which are {range_text}, it is clear that: ",
                    f"Taking into account the measured precipitation levels in dBZ, specifically {range_text}, we can conclude that: ",
                    f"The precipitation data, which shows dBZ values {range_text}, leads us to the following conclusion: ",
                    f"With the precipitation measurements in dBZ being {range_text}, it becomes evident that: ",
                    f"Upon analyzing the precipitation levels in dBZ, which are {range_text}, we can confidently state that: ",
                    f"Given the recorded precipitation measurements in dBZ, falling within {range_text}, we can determine that: ",
                    f"After examining the precipitation data with dBZ values {range_text}, we can affirm that: ",
                    f"By evaluating the precipitation measurements in dBZ, which are {range_text}, it is apparent that: ",
                    f"Through the assessment of the precipitation levels in dBZ, specifically {range_text}, we can deduce that: ",
                    f"Since the precipitation levels are measured in decibels relative to Z (dBZ), which represents the radar reflectivity, we can determine the intensity of the precipitation along the route. In this case, the dBZ values are {range_text}, leading us to conclude that: ",
                    f"By examining the dBZ values, which represent the radar reflectivity factor of precipitation, we can gain insights into the intensity and distribution of precipitation along the route. With dBZ values {range_text}, we can confidently determine that: ",
                    f"Taking into consideration the precipitation measurements in dBZ, a unit that quantifies radar reflectivity and directly relates to precipitation intensity, we can analyze the route's precipitation levels. With dBZ values {range_text}, it becomes clear that: ",
                    f"Given the precipitation data in dBZ, a metric that measures the radar reflectivity factor and provides insights into the intensity of precipitation, we can better understand the conditions along the route. The dBZ values are {range_text}, which leads us to the conclusion that: ",
                    f"When evaluating the precipitation levels, it's crucial to consider the dBZ values, as they represent the radar reflectivity factor and are directly linked to precipitation intensity. In this case, the dBZ values fall within {range_text}, allowing us to deduce that: ",
                    f"Assessing the dBZ values, which represent the radar reflectivity factor of precipitation, is essential in understanding the intensity and distribution of precipitation along a route. This information is crucial as it affects driving conditions, driver visibility, and overall safety. In this instance, the dBZ values are {range_text}, leading us to conclude that: ",
                    f"When analyzing precipitation levels, it's important to consider dBZ values, as they quantify radar reflectivity and provide insights into precipitation intensity. This data helps us understand how rain might impact driving conditions, driver visibility, and the overall route experience. With dBZ values {range_text}, we can confidently determine that: ",
                    f"Taking into account the precipitation measurements in dBZ, a unit that measures radar reflectivity and directly relates to precipitation intensity, we can evaluate the potential impact of rain on driving conditions, driver visibility, and the route itself. In this case, the dBZ values are {range_text}, which allows us to deduce that: ",
                    f"By examining precipitation data in dBZ, a metric that quantifies radar reflectivity and offers insights into the intensity of precipitation, we can better understand the potential effects of rain on driving conditions, driver visibility, and the route experience. In this situation, the dBZ values fall within {range_text}, leading us to the conclusion that: ",
                    f"Precipitation levels measured in dBZ, which represent the radar reflectivity factor, are crucial in determining the intensity and distribution of precipitation along a route. This information has a direct impact on driving conditions, driver visibility, and overall route safety. With dBZ values {range_text}, we can establish that: ",
                    f"To understand how rain might affect driving, we look at the dBZ values, which tell us how heavy the rain is. When we know how heavy the rain is, we can figure out how hard it might be to see while driving and how safe the route is. In this case, the dBZ values are {range_text}, which helps us figure out that: ",
                    f"When we talk about dBZ values, we're discussing how strong the rain is, which is important for understanding how it might impact drivers on a route. The heavier the rain, the more challenging it can be to drive safely and see clearly. With the dBZ values being {range_text}, we can say that: ",
                    f"Checking the dBZ values helps us know how intense the rain is along a route, which is important for drivers. Knowing the intensity of the rain can help us understand how it might affect visibility and the safety of the route. In this situation, the dBZ values are {range_text}, so we can conclude that: ",
                    f"Measuring rain intensity using dBZ values helps us understand how rain might influence driving conditions, the driver's ability to see, and the overall safety of a route. The higher the dBZ values, the heavier the rain. In this case, the dBZ values are {range_text}, which means: ",
                    f"When we look at dBZ values, we're trying to figure out how heavy the rain is along a route. This is important because it affects how easy it is to drive and see while driving. In this instance, the dBZ values are {range_text}, so we can determine that: "
                    f"The following tokens indicate {target_value}: {self.TOKEN_REASON_REPLACEMENT_TEXT}. ",
                    f"Indicators for {target_value} are: {self.TOKEN_REASON_REPLACEMENT_TEXT}. ",
                    f"These tokens signify {target_value}: {self.TOKEN_REASON_REPLACEMENT_TEXT}. ",
                ]
            )

        reply_list = [
            f"Yes, there is {target_value} consistently throughout the entire route.",
            f"Indeed, {target_value} can be found across the entire route.",
            f"Absolutely, {target_value} is present at all points along the route.",
            f"The answer is yes, as {target_value} remains consistent across the entire route.",
            f"You can be confident that {target_value} is present throughout the entire route.",
            f"Yes, the route includes {target_value} at all points.",
            f"The entire route consistently features {target_value}, so the answer is yes.",
            f"The answer is positive, as {target_value} is found throughout the entire route.",
            f"Yes, there is {target_value} consistently throughout the entire route. This means that the level of precipitation remains uniform across all sections, ensuring a predictable and stable travel experience.",
            f"Indeed, {target_value} can be found across the entire route, highlighting the fact that the precipitation level remains constant as you progress, providing consistent conditions during your journey.",
            f"The route contains {target_value} throughout, indicating that the level of precipitation is the same from start to finish. This uniformity offers a consistent travel experience with no surprises.",
            f"Absolutely, {target_value} is present at all points along the route. This signifies that the level of precipitation will be consistent during your entire journey, allowing for easier planning and preparation.",
            f"The answer is yes, as {target_value} remains consistent across the entire route. This suggests that the precipitation level will be uniform, offering a stable and predictable travel experience.",
            f"The entire route features {target_value}, which means the level of precipitation will remain the same as you travel. This consistency in conditions will make it easier to plan and prepare for your trip.",
            f"You can be confident that {target_value} is present throughout the entire route. This uniform level of precipitation ensures a consistent travel experience, allowing you to plan your journey accordingly.",
            f"Yes, the route includes {target_value} at all points, signifying that the precipitation level remains stable during the entire journey. This consistency allows for a more predictable experience for travelers.",
            f"The entire route consistently features {target_value}, so the answer is yes. This level of precipitation remains constant, allowing for a more predictable journey with no unexpected changes in conditions.",
            f"The answer is positive, as {target_value} is found throughout the entire route. This consistent level of precipitation ensures a stable travel experience, making it easier to plan and prepare for your journey.",
        ]

        return cot_pretext + random.choice(reply_list)

    def _get_specific_response_true_multi(
        self, segments, target_value, use_cot, position=None, plural=False
    ):
        cot_prepend_text = ""

        if use_cot:
            # get value range
            min_val = round(
                segments[segments[self.FEATURE_NAME] == target_value][
                    "precipitation_value"
                ].min()
            )
            max_val = round(
                segments[segments[self.FEATURE_NAME] == target_value][
                    "precipitation_value"
                ].max()
            )

            if min_val == max_val:
                range_text = random.choice(
                    [
                        f"constantly {min_val} dBZ",
                        f"constantly {min_val} dBZ",
                        f"{min_val} dBZ",
                        f"persistently {min_val} dBZ",
                    ]
                )
            else:
                range_text = random.choice(
                    [
                        f"in the range from {min_val} to {max_val} dBZ",
                        f"between {min_val} and {max_val} dBZ",
                        f"in the interval [{min_val}, {max_val}] dBZ",
                    ]
                )

            cot_prepend_text = random.choice(
                [
                    f"Based on the available data, the precipitation levels measured in dBZ are {range_text}. This indicates that the correct answer is: {'in multiple segments' if plural else 'in a single segment'}",
                    f"Considering the precipitation measurements in dBZ, which are {range_text}, it is clear that: {'the requested level of precipitation can be found in several sections' if plural else 'the requested level of precipitation is present in one section'}",
                    f"Taking into account the measured precipitation levels in dBZ, specifically {range_text}, we can conclude that: {'the target value occurs in various passages along the route' if plural else 'the target value occurs in a particular passage of the route'}",
                    f"The precipitation data, which shows dBZ values {range_text}, leads us to the following conclusion: {'the desired precipitation intensity is present in multiple parts of the route' if plural else 'the desired precipitation intensity is present in a single part of the route'}",
                    f"With the precipitation measurements in dBZ being {range_text}, it becomes evident that: {'the specified precipitation level is found in numerous areas along the route' if plural else 'the specified precipitation level is found in a single area along the route'}",
                    f"Upon analyzing the precipitation levels in dBZ, which are {range_text}, we can confidently state that: {'the requested precipitation is present in different portions of the route' if plural else 'the requested precipitation is present in a particular portion of the route'}",
                    f"Given the recorded precipitation measurements in dBZ, falling within {range_text}, we can determine that: {'the target precipitation is present in several stretches of the route' if plural else 'the target precipitation is present in one stretch of the route'}",
                    f"After examining the precipitation data with dBZ values {range_text}, we can affirm that: {'the desired precipitation level can be found in multiple segments of the route' if plural else 'the desired precipitation level can be found in a single segment of the route'}",
                    f"By evaluating the precipitation measurements in dBZ, which are {range_text}, it is apparent that: {'the target precipitation level is present in various sections throughout the route' if plural else 'the target precipitation level is present in a specific section of the route'}",
                    f"Through the assessment of the precipitation levels in dBZ, specifically {range_text}, we can deduce that: {'the requested precipitation level can be found in multiple parts of the route' if plural else 'the requested precipitation level can be found in one part of the route'}",
                    f"Since the precipitation levels are measured in decibels relative to Z (dBZ), which represents the radar reflectivity, we can determine the intensity of the precipitation along the route. In this case, the dBZ values are {range_text}, leading us to conclude that: {'the target precipitation is present in multiple zones of the route' if plural else 'the target precipitation is present in a single zone of the route'}",
                    f"By examining the dBZ values, which represent the radar reflectivity factor of precipitation, we can gain insights into the intensity and distribution of precipitation along the route. With dBZ values {range_text}, we can confidently determine that: {'the specified precipitation level is encountered in various intervals along the route' if plural else 'the specified precipitation level is encountered in a specific interval along the route'}",
                    f"Taking into consideration the precipitation measurements in dBZ, a unit that quantifies radar reflectivity and directly relates to precipitation intensity, we can analyze the route's precipitation levels. With dBZ values {range_text}, it becomes clear that: {'the requested precipitation level is present in multiple stretches along the route' if plural else 'the requested precipitation level is present in a single stretch along the route'}",
                    f"Given the precipitation data in dBZ, a metric that measures the radar reflectivity factor and provides insights into the intensity of precipitation, we can better understand the conditions along the route. The dBZ values are {range_text}, which leads us to the conclusion that: {'the target precipitation level can be found in numerous sections throughout the route' if plural else 'the target precipitation level can be found in a particular section of the route'}",
                    f"When evaluating the precipitation levels, it's crucial to consider the dBZ values, as they represent the radar reflectivity factor and are directly linked to precipitation intensity. In this case, the dBZ values fall within {range_text}, allowing us to deduce that: {'the requested precipitation level is present in various parts of the route' if plural else 'the requested precipitation level is present in one part of the route'}",
                    f"Assessing the dBZ values, which represent the radar reflectivity factor of precipitation, is essential in understanding the intensity and distribution of precipitation along a route. This information is crucial as it affects driving conditions, driver visibility, and overall safety. In this instance, the dBZ values are {range_text}, leading us to conclude that: {'the desired precipitation level can be found in multiple areas of the route' if plural else 'the desired precipitation level can be found in a single area of the route'}",
                    f"When analyzing precipitation levels, it's important to consider dBZ values, as they quantify radar reflectivity and provide insights into precipitation intensity. This data helps us understand how rain might impact driving conditions, driver visibility, and the overall route experience. With dBZ values {range_text}, we can confidently determine that: {'the target precipitation level is present in several segments along the route' if plural else 'the target precipitation level is present in a single segment of the route'}",
                    f"Taking into account the precipitation measurements in dBZ, a unit that measures radar reflectivity and directly relates to precipitation intensity, we can evaluate the potential impact of rain on driving conditions, driver visibility, and the route itself. In this case, the dBZ values are {range_text}, which allows us to deduce that: {'the specified precipitation level is found in multiple passages of the route' if plural else 'the specified precipitation level is found in a single passage of the route'}",
                    f"By examining precipitation data in dBZ, a metric that quantifies radar reflectivity and offers insights into the intensity of precipitation, we can better understand the potential effects of rain on driving conditions, driver visibility, and the route experience. In this situation, the dBZ values fall within {range_text}, leading us to the conclusion that: {'the requested precipitation level is present in various sections throughout the route' if plural else 'the requested precipitation level is present in a specific section of the route'}",
                    f"Precipitation levels measured in dBZ, which represent the radar reflectivity factor, are crucial in determining the intensity and distribution of precipitation along a route. This information has a direct impact on driving conditions, driver visibility, and overall route safety. With dBZ values {range_text}, we can establish that: {'the desired precipitation level is present in multiple portions of the route' if plural else 'the desired precipitation level is present in a single portion of the route'}",
                    f"To understand how rain might affect driving, we look at the dBZ values, which tell us how heavy the rain is. When we know how heavy the rain is, we can figure out how hard it might be to see while driving and how safe the route is. In this case, the dBZ values are {range_text}, which helps us figure out that: {'the target precipitation level is present in several intervals along the route' if plural else 'the target precipitation level is present in a specific interval along the route'}",
                    f"When we talk about dBZ values, we're discussing how strong the rain is, which is important for understanding how it might impact drivers on a route. The heavier the rain, the more challenging it can be to drive safely and see clearly. With the dBZ values being {range_text}, we can say that: {'the requested precipitation level is encountered in multiple stretches of the route' if plural else 'the requested precipitation level is encountered in one stretch of the route'}",
                    f"Checking the dBZ values helps us know how intense the rain is along a route, which is important for drivers. Knowing the intensity of the rain can help us understand how it might affect visibility and the safety of the route. In this situation, the dBZ values are {range_text}, so we can conclude that: {'the specified precipitation level is present in various zones of the route' if plural else 'the specified precipitation level is present in a single zone of the route'}",
                    f"Measuring rain intensity using dBZ values helps us understand how rain might influence driving conditions, the driver's ability to see, and the overall safety of a route. The higher the dBZ values, the heavier the rain. In this case, the dBZ values are {range_text}, which means: {'the desired precipitation level can be found in numerous areas along the route' if plural else 'the desired precipitation level can be found in a single area along the route'}",
                    f"When we look at dBZ values, we're trying to figure out how heavy the rain is along a route. This is important because it affects how easy it is to drive and see while driving. In this instance, the dBZ values are {range_text}, so we can determine that: {'the target precipitation level is present in multiple parts of the route' if plural else 'the target precipitation level is present in one part of the route'}",
                    f"The following tokens indicate {target_value}: {self.TOKEN_REASON_REPLACEMENT_TEXT}. ",
                    f"Indicators for {target_value} are: {self.TOKEN_REASON_REPLACEMENT_TEXT}. ",
                    f"These tokens signify {target_value}: {self.TOKEN_REASON_REPLACEMENT_TEXT}. ",
                ]
            )

        if position is not None:
            if plural:
                reply_list = [
                    f"Yes, {position} of the route",
                    f"Yes, there is {target_value} consistently throughout multiple passages {position} of the trip.",
                    f"Indeed, {target_value} can be found multiple times {position} of the journey.",
                    f"Absolutely, {target_value} is present at several points {position}.",
                    f"The answer is yes, as {target_value} remains consistent across multiple segments {position} of the trip.",
                    f"You can be confident that {target_value} is present several times {position}.",
                    f"Yes, the route includes {target_value} at several points {position}.",
                    f"The route features {target_value} multiple times {position}, so the answer is yes.",
                    f"The answer is positive, as {target_value} is found at different points {position} of the route.",
                ]
            else:
                reply_list = [
                    f"Yes, {position} of the route",
                    f"Yes, there is {target_value} consistently throughout one passage {position} of the trip.",
                    f"Indeed, {target_value} can be found one time {position} of the journey.",
                    f"Absolutely, {target_value} is present at one point {position}.",
                    f"The answer is yes, as {target_value} remains consistent across once section {position} of the trip.",
                    f"You can be confident that {target_value} is present once {position}.",
                    f"Yes, the route includes {target_value} at one point {position}.",
                    f"The route features {target_value} once {position}, so the answer is yes.",
                    f"The answer is positive, as {target_value} is found at one point {position} of the route.",
                ]

        else:
            if plural:
                reply_list = [
                    "Yes",
                    f"Yes, there is {target_value} consistently throughout multiple passages of the route.",
                    f"Indeed, {target_value} can be found multiple times across the entire route.",
                    f"Absolutely, {target_value} is present at several points along the route.",
                    f"The answer is yes, as {target_value} remains consistent across multiple segments of the trip.",
                    f"You can be confident that {target_value} is present several times throughout the journey.",
                    f"Yes, the route includes {target_value} at several points.",
                    f"The entire route features {target_value} multiple times, so the answer is yes.",
                    f"The answer is positive, as {target_value} is found at different points throughout the entire route.",
                ]
            else:
                reply_list = [
                    "Yes",
                    f"Yes, there is {target_value} in one passage of the route.",
                    f"Indeed, {target_value} can be found only once across the entire route.",
                    f"Absolutely, {target_value} is present at a single point along the route.",
                    f"The answer is yes, as {target_value} remains consistent across one section of the trip.",
                    f"You can be confident that {target_value} is present in the journey.",
                    f"Yes, the route includes {target_value} at one point.",
                    f"The entire route features {target_value} in one section, so the answer is yes.",
                    f"The answer is positive, as {target_value} is found at one point of the route.",
                ]

        return cot_prepend_text + random.choice(reply_list)

    def _get_specific_response_false(self, segments, target_value, use_cot):
        cot_prepend_text = ""

        if use_cot:
            # get value range
            min_val = round(segments["precipitation_value"].min())
            max_val = round(segments["precipitation_value"].max())

            if min_val == max_val:
                range_text = random.choice(
                    [
                        f"constantly {min_val} dBZ",
                        f"constantly {min_val} dBZ",
                        f"{min_val} dBZ",
                        f"persistently {min_val} dBZ",
                    ]
                )
            else:
                range_text = random.choice(
                    [
                        f"in the range from {min_val} to {max_val} dBZ",
                        f"between {min_val} and {max_val} dBZ",
                        f"in the interval [{min_val}, {max_val}] dBZ",
                    ]
                )

            cot_prepend_text = random.choice(
                [
                    f"Based on the available data, the precipitation levels measured in dBZ are {range_text}. This indicates that the target value {target_value} is not present in the route: ",
                    f"Considering the precipitation measurements in dBZ, which are {range_text}, it is clear that the target value {target_value} is not present in the route: ",
                    f"Taking into account the measured precipitation levels in dBZ, specifically {range_text}, we can conclude that the target value {target_value} is not present in the route: ",
                    f"The precipitation data, which shows dBZ values {range_text}, leads us to the following conclusion: the target value {target_value} is not present in the route.",
                    f"With the precipitation measurements in dBZ being {range_text}, it becomes evident that the target value {target_value} is not present in the route: ",
                    f"Upon analyzing the precipitation levels in dBZ, which are {range_text}, we can confidently state that the target value {target_value} is not present in the route: ",
                    f"Given the recorded precipitation measurements in dBZ, falling within {range_text}, we can determine that the target value {target_value} is not present in the route: ",
                    f"After examining the precipitation data with dBZ values {range_text}, we can affirm that the target value {target_value} is not present in the route: ",
                    f"By evaluating the precipitation measurements in dBZ, which are {range_text}, it is apparent that the target value {target_value} is not present in the route: ",
                    f"Through the assessment of the precipitation levels in dBZ, specifically {range_text}, we can deduce that the target value {target_value} is not present in the route: ",
                    f"Since the precipitation levels are measured in decibels relative to Z (dBZ), which represents the radar reflectivity, we can determine the intensity of the precipitation along the route. In this case, the dBZ values are {range_text}, leading us to conclude that the target value {target_value} is not present in the route.",
                    f"By examining the dBZ values, which represent the radar reflectivity factor of precipitation, we can gain insights into the intensity and distribution of precipitation along the route. With dBZ values {range_text}, we can confidently determine that the target value {target_value} is not present in the route.",
                    f"Taking into consideration the precipitation measurements in dBZ, a unit that quantifies radar reflectivity and directly relates to precipitation intensity, we can analyze the route's precipitation levels. With dBZ values {range_text}, it becomes clear that the target value {target_value} is not present in the route.",
                    f"Given the precipitation data in dBZ, a metric that measures the radar reflectivity factor and provides insights into the intensity of precipitation, we can better understand the conditions along the route. The dBZ values are {range_text}, which leads us to the conclusion that the target value {target_value} is not present in the route.",
                    f"When evaluating the precipitation levels, it's crucial to consider the dBZ values, as they represent the radar reflectivity factor and are directly linked to precipitation intensity. In this case, the dBZ values fall within {range_text}, allowing us to deduce that the target value {target_value} is not present in the route.",
                    f"Assessing the dBZ values, which represent the radar reflectivity factor of precipitation, is essential in understanding the intensity and distribution of precipitation along a route. This information is crucial as it affects driving conditions, driver visibility, and overall safety. In this instance, the dBZ values are {range_text}, leading us to conclude that the target value {target_value} is not present in the route.",
                    f"When analyzing precipitation levels, it's important to consider dBZ values, as they quantify radar reflectivity and provide insights into precipitation intensity. This data helps us understand how rain might impact driving conditions, driver visibility, and the overall route experience. With dBZ values {range_text}, we can confidently determine that the target value {target_value} is not present in the route.",
                    f"Taking into account the precipitation measurements in dBZ, a unit that measures radar reflectivity and directly relates to precipitation intensity, we can evaluate the potential impact of rain on driving conditions, driver visibility, and the route itself. In this case, the dBZ values are {range_text}, which allows us to deduce that the target value {target_value} is not present in the route.",
                    f"By examining precipitation data in dBZ, a metric that quantifies radar reflectivity and offers insights into the intensity of precipitation, we can better understand the potential effects of rain on driving conditions, driver visibility, and the route experience. In this situation, the dBZ values fall within {range_text}, leading us to the conclusion that the target value {target_value} is not present in the route.",
                    f"Precipitation levels measured in dBZ, which represent the radar reflectivity factor, are crucial in determining the intensity and distribution of precipitation along a route. This information has a direct impact on driving conditions, driver visibility, and overall route safety. With dBZ values {range_text}, we can establish that the target value {target_value} is not present in the route.",
                    f"To understand how rain might affect driving, we look at the dBZ values, which tell us how heavy the rain is. When we know how heavy the rain is, we can figure out how hard it might be to see while driving and how safe the route is. In this case, the dBZ values are {range_text}, which helps us figure out that the target value {target_value} is not present in the route.",
                    f"When we talk about dBZ values, we're discussing how strong the rain is, which is important for understanding how it might impact drivers on a route. The heavier the rain, the more challenging it can be to drive safely and see clearly. With the dBZ values being {range_text}, we can say that the target value {target_value} is not present in the route.",
                    f"Checking the dBZ values helps us know how intense the rain is along a route, which is important for drivers. Knowing the intensity of the rain can help us understand how it might affect visibility and the safety of the route. In this situation, the dBZ values are {range_text}, so we can conclude that the target value {target_value} is not present in the route.",
                    f"Measuring rain intensity using dBZ values helps us understand how rain might influence driving conditions, the driver's ability to see, and the overall safety of a route. The higher the dBZ values, the heavier the rain. In this case, the dBZ values are {range_text}, which means the target value {target_value} is not present in the route.",
                    self._categories_to_text(
                        segments[self.FEATURE_NAME].unique().tolist()
                    ),
                ]
            )

        reply_list = [
            f"No, at no point in the route there is {target_value}.",
            "No",
            f"There will be no {target_value} on this journey.",
            f"It appears that at no point along your route will you encounter {target_value}, as the precipitation levels differ from what you're looking for.",
            f"The current weather forecast shows that {target_value} is not present throughout your route, so you can expect other precipitation levels instead.",
            f"As you travel along your route, you won't come across any instances of {target_value}, meaning the precipitation levels will be different from that specific condition.",
            f"Based on the weather information, your route doesn't have any occurrences of {target_value}, so be prepared for other levels of precipitation along the way.",
            f"It seems that {target_value} is not a concern for your route, as that particular level of precipitation is not present at any point. Keep an eye out for other precipitation conditions instead.",
            f"Upon reviewing the weather forecast for your route, it has been determined that {target_value} is not present at any point during your journey. This means that you can expect to encounter different precipitation levels, which may require you to adjust your plans or preparations accordingly. Be sure to keep an eye on the weather conditions and stay prepared for any changes that may occur as you make your way along your planned path.",
            f"As you set out on your journey, it's important to note that the specific level of precipitation you're concerned about, {target_value}, is not present anywhere along your route. Instead, you'll come across varying precipitation levels, which could impact the overall experience of your trip. To ensure a smooth journey, it's crucial to stay informed about the changing weather conditions and adjust your plans as needed to accommodate any unexpected shifts in precipitation.",
            f"Based on the available weather data, it's clear that {target_value} won't be a factor throughout your route, as that particular level of precipitation is not present at any point during your travels. This means that you'll likely encounter different weather conditions, which could range from lighter to heavier precipitation or even a lack thereof. In light of this information, it's important to remain vigilant about the weather updates and make any necessary adjustments to your plans to ensure a safe and enjoyable journey.",
            f"Hey there! Just a heads up that {target_value} isn't going to be a thing on your route. So, you'll come across different types of rain or maybe no rain at all. Keep an eye on the weather and be ready to change your plans if needed. Have a great trip!",
            f"You won't find any {target_value} along your route. That means you'll experience other types of precipitation or even have clear skies. Just remember to stay aware of the weather and make any changes to your plans if necessary. Enjoy your journey!",
        ]

        return cot_prepend_text + random.choice(reply_list)

    def _categories_to_text(self, unique_categories, position=None):
        if len(unique_categories) == 1:
            rain_level = unique_categories[0]
            if position is None:
                reply_list = [
                    f"There is {rain_level} consistently throughout the entire route.",
                    f"{rain_level.capitalize()} can be found across the entire route.",
                    f"{rain_level.capitalize()} is present at all points along the route.",
                    f"{rain_level.capitalize()} remains consistent across the entire route.",
                    f"{rain_level.capitalize()} is present throughout the entire route.",
                    f"The route includes {rain_level} at all points.",
                    f"The entire route consistently features {rain_level}.",
                    f"{rain_level.capitalize()} is found throughout the entire route.",
                ]
            else:
                reply_list = [
                    f"There is {rain_level} {position} of the route.",
                    f"{rain_level.capitalize()} can be found {position} of the route.",
                    f"{rain_level.capitalize()} is present {position} of the route.",
                    f"{rain_level.capitalize()} remains consistent {position} of the route.",
                    f"{rain_level.capitalize()} is present {position} of the route.",
                    f"The route includes {rain_level} {position}.",
                    f"The route consistently features {rain_level} {position}.",
                    f"{rain_level.capitalize()} is found throughout {position}.",
                ]

            return random.choice(reply_list)

        rain_values = np.array(
            [PRECIPITATION_LIMITS[rain_value] for rain_value in unique_categories]
        )
        rain_values[unique_categories == "very heavy rain"] = 70.0

        min_dbz = unique_categories[rain_values.argmin()]
        max_dbz = unique_categories[rain_values.argmax()]

        if position is None:
            reply_list = [
                f"The precipitation level ranges from {min_dbz} to {max_dbz}.",
                f"During this trip there will be {min_dbz} to {max_dbz}.",
                f"On this journey you will experience {min_dbz} to {max_dbz}.",
            ]
        else:
            reply_list = [
                f"The precipitation level ranges from {min_dbz} to {max_dbz} {position}.",
                f"{position.capitalize()} there will be {min_dbz} to {max_dbz}.",
                f"{position.capitalize()} of this journey you will experience {min_dbz} to {max_dbz}.",
            ]

        return random.choice(reply_list)

    def pre_process_df(self, route_df):
        feature_names = route_df[self.FEATURE_NAME].apply(convert_precipitation_to_word)
        route_df["precipitation_value"] = route_df[self.FEATURE_NAME]
        route_df[self.FEATURE_NAME] = feature_names
        return route_df
