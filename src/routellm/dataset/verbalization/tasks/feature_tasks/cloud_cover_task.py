import random

import numpy as np
import pandas as pd

from routellm.dataset.routes.const import (
    CLOUD_COVER_LIMITS,
    convert_cloud_cover_to_word,
)
from routellm.dataset.verbalization.tasks.categorical_base_task import (
    CategoricalBaseTask,
)


class CloudCoverTask(CategoricalBaseTask):
    NAME = "feature-tasks.cloud-cover-task"
    VERSION = 0
    FEATURE_NAME = "cloudCover"
    CATEGORIES = [*CLOUD_COVER_LIMITS.keys()]

    def _get_general_task(
        self,
    )-> str:
        """
        Get a general task description for the cloud cover feature.

        Returns:
            str: A randomly selected task description related to cloud cover.

        """
        task_list = [
            *[
                f"What is the cloud coverage during this {word}"
                for word in ["route", "journey", "trip"]
            ],
            "Can you provide an overview of the cloud coverage",
            "How cloudy is the sky on this route",
            "Are there any clouds during the journey",
            "How cloudy will the sky be on this trip",
            "I will be driving the route at hand and I was wondering, what the cloud coverage will be like",
        ]

        return random.choice(task_list)

    def _get_specific_value_task(self, target_value: str) -> str:
        """
        Get a specific task description for the cloud cover feature based on the target value.

        Args:
            target_value (str): The target value for cloud cover, e.g., "clear", "partly cloudy", etc.

        Returns:
            str: A randomly selected task description related to the specific cloud cover value.

        """
        task_list = [
            f"Is the sky {target_value}",
            f"While passing through this route, are there any instances, where the sky is {target_value}",
            f"I am driving on this route and I was wondering if the sky will be {target_value}",
            f"{target_value.capitalize()} sky",
            f"{target_value} skies",
            f"Any {target_value} skies",
            f"Will the sky be {target_value}",
            f"This will be the journey I am going on today. I just wanted to make sure whether the sky will be {target_value}",
            f"Is the weather {target_value}",
        ]

        return random.choice(task_list)

    def _get_specific_response_true_single(self, segments: pd.DataFrame, target_value: str, use_cot: bool) -> str:
        """
        Generate a response for a single segment route based on the cloud cover value.

        Args:
            segments (pd.DataFrame): DataFrame containing the segments with cloud cover values.
            target_value (str): The target cloud cover value, e.g., "clear", "partly cloudy", etc.
            use_cot (bool): Whether to use chain-of-thought reasoning in the response.

        Returns:
            str: A randomly selected response template indicating the presence of the target cloud cover value.

        """
        if use_cot:
            # get value range
            min_val = round(segments["cloudCover_value"].min())
            max_val = round(segments["cloudCover_value"].max())

            if min_val == max_val:
                range_text = random.choice(
                    [
                        f"constantly {min_val} %",
                        f"constantly {min_val} percent",
                        f"{min_val} %",
                        f"persistently {min_val} percent",
                    ]
                )
            else:
                range_text = random.choice(
                    [
                        f"in the range from {min_val} to {max_val} percent",
                        f"between {min_val} and {max_val} %",
                        f"in the interval [{min_val}, {max_val}] percent",
                    ]
                )

            reply_list = [
                f"Thank you for your request. In the data you provided the cloud coverage is {range_text}, therefore, yes, the sky will be {target_value} for the whole journey.",
                f"The cloud coverage on this route is {range_text}. Yes, the sky will be {target_value} during the whole trip.",
                f"In the given route, the cloud coverage is {range_text}, meaning that the sky is {target_value}.",
                f"Due to the coverage being {range_text}, the sky is {target_value} during the route.",
                f"With the coverage measurements being {range_text}, the weather is {target_value}.",
                f"You are asking about the cloud coverage on the given route. The cloud coverage is the percentage of the sky that is obstructed by clouds. The cloud coverage value is {range_text} for the whole journey. The answer is yes, the sky is {target_value}.",
                f"The following tokens indicate that the sky is {target_value}: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                f"Indicators for a {target_value} sky are: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
                f"These tokens signify a {target_value} sky: {self.TOKEN_REASON_REPLACEMENT_TEXT}.",
            ]
        else:
            reply_list = [
                f"Thank you for your request. As of the provided data, yes, the sky will be {target_value} for the whole journey.",
                f"Yes, the sky will be {target_value} during the whole trip.",
                f"In the given route, the cloud coverage is {target_value}.",
                f"Yes, the sky is {target_value} during the route.",
                f"Indeed, the weather is {target_value}.",
                f"The whole journey takes place, while the sky is {target_value}.",
            ]

        return random.choice(reply_list)

    def _get_specific_response_true_multi(
        self, segments: pd.DataFrame, target_value: str, use_cot: bool, position: str | None=None, plural: bool=False
    ) -> str:
        """
        Generate a response for multi-segment routes based on the cloud cover value.

        Args:
            segments (pd.DataFrame): DataFrame containing the segments with cloud cover values.
            target_value (str): The target cloud cover value, e.g., "clear", "partly cloudy", etc.
            use_cot (bool): Whether to use chain-of-thought reasoning in the response.
            position (str | None, optional): Position information for the segment(s). Defaults to None.
            plural (bool, optional): Whether there are multiple matching segments. Defaults to False.

        Returns:
            str: A randomly selected response template indicating the presence of the target cloud cover value in multiple segments.

        """
        cot_prepend_text = ""

        if use_cot:
            # get value range
            min_val = round(
                segments[segments[self.FEATURE_NAME] == target_value][
                    "cloudCover_value"
                ].min()
            )
            max_val = round(
                segments[segments[self.FEATURE_NAME] == target_value][
                    "cloudCover_value"
                ].max()
            )

            if min_val == max_val:
                insert_text = random.choice([f"{min_val} %", f"{min_val} percent"])
            else:
                insert_text = random.choice(
                    [
                        f"{min_val} to {max_val} %",
                        f"{min_val} to {max_val} percent",
                    ]
                )

            cot_prepend_text = random.choice(
                [
                    f"You are asking about the cloud coverage in this route. The term refers to the amount of clouds present in the sky at a given time. It is often expressed as a percentage, indicating how much of the sky is covered by clouds.\n\nIn the given route there {'are multiple segments' if plural else 'is a segment'} with a coverage of {insert_text}. The resulting coverage can be described as {target_value}.\n\nThe correct answer is: ",
                    f"The request aims at understanding the cloud coverage on the provided trip.\n\nCloud coverage is how much of the sky is filled with clouds. It's like a blanket that can be thin or thick, covering parts of the sky or all of it.\n\n In the example, part{'s' if plural else ''} of the route {'have' if plural else 'has'} a coverage of {insert_text}.\n\nThis brings me to my answer: ",
                    f"To answer this question, it is necessary to understand the term cloud coverage. It refers to the amount of clouds in the sky at a given time. It can affect the weather, like making it cooler or causing rain, and is measured as a percentage of the sky covered by clouds.\n\nThe provided route has {'passages' if plural else 'a passage'} with a coverage of {insert_text} ({target_value}).\n\nThe answer: ",
                    f"The question can be answered by measuring the cloud coverage along the route. Cloud coverage is the proportion of the sky obscured by clouds, which can impact temperature, precipitation, and sunlight. It plays a significant role in weather forecasting and is expressed as a percentage or in eighths (oktas).\n\nThe described journey has {'multiple instances' if plural else 'one instance'} where the measurement is {insert_text}, meaning the weather is {target_value}.\n\nThis concludes my answer: ",
                    f"The question can easily be answered with the respective cloud coverage of the journey. Cloud coverage describes the fraction of the sky covered by clouds, affecting atmospheric processes such as solar radiation and precipitation. Accurately predicting cloud coverage requires sophisticated numerical models that consider various factors, including humidity, temperature, air movement, and aerosol-cloud interactions.\n\nThe provided path has {'multiple occasions' if plural else 'one occasion'} where the measurement is {insert_text}, meaning the sky is {target_value}.\n\nAnswer: ",
                    f"An answer to this question can be found by checking the cloud coverage. Cloud coverage means how many clouds are in the sky. It changes weather and sunlight. We say it as a percent or parts of eight.\n\nIn the given route there {'are multiple segments' if plural else 'is a segment'} with a coverage of {insert_text}. The resulting coverage can be described as {target_value}.\n\nThe correct answer is: ",
                    f"Your question can be answered by inspecting the cloud coverage at several points in the journey. Cloud coverage tells us the amount of sky that has clouds in it. Imagine the sky as a big painting, and the clouds are the parts with paint on them.\n\n In the example, part{'s' if plural else ''} of the route {'have' if plural else 'has'} a coverage of {insert_text}.\n\nThis brings me to my answer: ",
                    f"You are asking about the cloud coverage on this route. Cloud coverage is the measurement of clouds occupying the sky. It influences weather conditions like temperature and rainfall and is shown as a percentage to indicate how much of the sky has clouds.\n\nIn the given route there {'are multiple segments' if plural else 'is a segment'} with a coverage of {insert_text}. The resulting coverage can be described as {target_value}.\n\nThe correct answer is: ",
                    f"In this case cloud coverage denotes the extent to which clouds obscure the sky, impacting various weather factors such as sunlight, temperature, and precipitation. It is commonly expressed as a percentage or in eighths, called oktas.\n\nThe provided route has {'passages' if plural else 'a passage'} with a coverage of {insert_text} ({target_value}).\n\nThe answer: ",
                    f"Cloud coverage pertains to the fraction of the sky obscured by clouds, which influences atmospheric phenomena like solar radiation and precipitation. Precise cloud coverage predictions necessitate advanced numerical models that account for variables such as humidity, air temperature, wind patterns, and aerosol-cloud interactions.\n\nIn the route at hand there {'are multiple segments' if plural else 'is a segment'} with a coverage of {insert_text}. The resulting coverage can be described as {target_value}.\n\nThe correct answer is: ",
                    f"Cloud coverage describes the percentage of the sky that is covered by clouds. In the given route there {'are multiple segments' if plural else 'is a segment'} with a coverage of {insert_text}. The resulting coverage can be described as {target_value}.\n\n",
                    f"The question hints at the cloud coverage meaning the fraction of the sky that is obstructed by clouds. The route has {'at least ' if plural else ''}one segment where the cloud coverage is {insert_text}, meaning the sky is {target_value}.\nThe answer is: ",
                    f"One way to describe the weather is by describing how clear/cloudy the sky is. This can be measured using the cloud coverage.\n\nIn simple words: The cloud coverage is how much of the sky is covered by clouds. The value is measured in percent, where 0% are no clouds and 100% would mean a fully cloudy sky.\nThe value can also be described in words. Often, a clear sky references a sky without no clouds, whereas an overcast sky is covered in clouds. There are more steps between clear and cloudy, i.e. mostly clear, partly cloudy and many more.\n\nThe route you provided has {'at least ' if plural else ''}one segment where the cloud coverage is {insert_text}. In words this would mean the sky is {target_value}.\nTherefore the answer is: ",
                    f"Cloud coverage, also known as cloud cover or cloudiness, is a meteorological term that refers to the fraction of the sky obscured by clouds when observed from a particular location. It is typically expressed as a percentage or as a fraction of eighths (oktas). Cloud coverage plays a crucial role in various atmospheric processes, including the regulation of temperature, precipitation, and solar radiation.\n\nUnderstanding and predicting cloud coverage is essential for meteorologists, climatologists, and atmospheric scientists, as it impacts weather forecasting, climate modeling, and renewable energy resource assessments.\n\nThere {'are segments' if plural else 'is a segment'} in the given route, where the cloud coverage is {insert_text}. This can be described as {target_value}.\n\nThis results in the following answer: ",
                    f"The following tokens indicate that the sky is {target_value}: {self.TOKEN_REASON_REPLACEMENT_TEXT}. ",
                    f"Indicators for a {target_value} sky are: {self.TOKEN_REASON_REPLACEMENT_TEXT}. ",
                    f"These tokens signify a {target_value} sky: {self.TOKEN_REASON_REPLACEMENT_TEXT}. ",
                ]
            )

        if position is not None:
            if plural:
                reply_list = [
                    f"Yes there are multiple instances of the weather being {target_value} {position}.",
                    f"While travelling on this route, there are multiple sections {position} where the sky is {target_value}.",
                    f"In some segments {position} the sky is {target_value}.",
                    f"Hey, I checked the route you provided and found multiple occasions {position} that match your description.",
                ]
            else:
                reply_list = [
                    f"Yes there is an instance of the weather being {target_value} {position}.",
                    f"Hey, I checked the route you provided and found exactly one occasion {position} that matches your description.",
                    f"The weather is {target_value} {position}.",
                    f"Yes {position} the sky is {target_value}.",
                ]
        else:
            if plural:
                reply_list = [
                    f"There are multiple occasions where the sky is {target_value}.",
                    f"In more than one section is the weather {target_value}.",
                    f"Yes, {target_value} weather is present multiple times on this route.",
                    "Hey, I checked the route you provided and found multiple occasions that match your description.",
                    f"Yes at multiple points during the drive the weather is {target_value}.",
                ]
            else:
                reply_list = [
                    f"There is one occasion where the weather is {target_value}.",
                    "Yes",
                    f"Given your description I found one passage in the route, where the sky is {target_value}.",
                    "Hey, I checked the route you provided and found exactly one occasion that matches your description.",
                    f"Yes at one part of the route the sky is {target_value}.",
                ]

        return cot_prepend_text + random.choice(reply_list)

    def _get_specific_response_false(self, segments: pd.DataFrame, target_value: str, use_cot: bool) -> str:
        """
        Generate a response indicating that the target cloud cover value is not present in the segments.

        Args:
            segments (pd.DataFrame): DataFrame containing the segments with cloud cover values.
            target_value (str): The target cloud cover value, e.g., "clear", "partly cloudy", etc.
            use_cot (bool): Whether to use chain-of-thought reasoning in the response.

        Returns:
            str: A randomly selected response template indicating that the target cloud cover value is not present in the segments.

        """
        cot_prepend_text = ""

        if use_cot:
            cot_prepend_text = random.choice(
                [
                    "You are asking about the cloud coverage in this route. The term refers to the amount of clouds present in the sky at a given time. It is often expressed as a percentage, indicating how much of the sky is covered by clouds.\n\nThe correct answer is: ",
                    "The request aims at understanding the cloud coverage on the provided trip.\n\nCloud coverage is how much of the sky is filled with clouds. It's like a blanket that can be thin or thick, covering parts of the sky or all of it.\n\nThis brings me to my answer: ",
                    "To answer this question, it is necessary to understand the term cloud coverage. It refers to the amount of clouds in the sky at a given time. It can affect the weather, like making it cooler or causing rain, and is measured as a percentage of the sky covered by clouds.\n\nThe answer: ",
                    "The question can be answered by measuring the cloud coverage along the route. Cloud coverage is the proportion of the sky obscured by clouds, which can impact temperature, precipitation, and sunlight. It plays a significant role in weather forecasting and is expressed as a percentage or in eighths (oktas).\n\nThis concludes my answer: ",
                    "The question can easily be answered with the respective cloud coverage of the journey. Cloud coverage describes the fraction of the sky covered by clouds, affecting atmospheric processes such as solar radiation and precipitation. Accurately predicting cloud coverage requires sophisticated numerical models that consider various factors, including humidity, temperature, air movement, and aerosol-cloud interactions.\n\nAnswer: ",
                    "An answer to this question can be found by checking the cloud coverage. Cloud coverage means how many clouds are in the sky. It changes weather and sunlight. We say it as a percent or parts of eight.\n\nThe correct answer is: ",
                    "Your question can be answered by inspecting the cloud coverage at several points in the journey. Cloud coverage tells us the amount of sky that has clouds in it. Imagine the sky as a big painting, and the clouds are the parts with paint on them.\n\nThis brings me to my answer: ",
                    "You are asking about the cloud coverage on this route. Cloud coverage is the measurement of clouds occupying the sky. It influences weather conditions like temperature and rainfall and is shown as a percentage to indicate how much of the sky has clouds.\n\nThe correct answer is: ",
                    "In this case cloud coverage denotes the extent to which clouds obscure the sky, impacting various weather factors such as sunlight, temperature, and precipitation. It is commonly expressed as a percentage or in eighths, called oktas.\n\nThe answer: ",
                    "Cloud coverage pertains to the fraction of the sky obscured by clouds, which influences atmospheric phenomena like solar radiation and precipitation. Precise cloud coverage predictions necessitate advanced numerical models that account for variables such as humidity, air temperature, wind patterns, and aerosol-cloud interactions.\n\nThe correct answer is: ",
                    "Cloud coverage describes the percentage of the sky that is covered by clouds.\n",
                    "The question hints at the cloud coverage meaning the fraction of the sky that is obstructed by clouds.\nThe answer is: ",
                    "One way to describe the weather is by describing how clear/cloudy the sky is. This can be measured using the cloud coverage.\n\nIn simple words: The cloud coverage is how much of the sky is covered by clouds. The value is measured in percent, where 0% are no clouds and 100% would mean a fully cloudy sky.\nThe value can also be described in words. Often, a clear sky references a sky without no clouds, whereas an overcast sky is covered in clouds. There are more steps between clear and cloudy, i.e. mostly clear, partly cloudy and many more.\nTherefore the answer is: ",
                ]
            )

        reply_list = [
            f"No, there are no passages in this journey, where the weather is {target_value}.",
            f"The sky will not be {target_value}.",
            f"No, the sky is not {target_value}.",
            f"No, at no point in the trip will the weather be {target_value}.",
            f"Hi, I reviewed your request. I have to inform you that given the route in your message there are no sections with {target_value} skies.",
            f"For the whole route the sky is not {target_value}.",
            f"There is no {target_value} weather or indications for it in the provided route.",
        ]

        return cot_prepend_text + random.choice(reply_list)

    def _categories_to_text(self, unique_categories: list[str], position: str | None=None) -> str:
        """
        Convert unique categories of cloud cover into a descriptive text.

        Args:
            unique_categories (list[str]): List of unique cloud cover categories.
            position (str | None, optional): Position information for the segment(s). Defaults to None

        Returns:
            str: A descriptive text summarizing the cloud cover categories.

        """
        if len(unique_categories) == 1:
            coverage = unique_categories[0]
            if position is None:
                reply_list = [
                    f"The sky is {coverage} during the whole journey.",
                    f"During the entirety of the route the sky is {coverage}.",
                ]
            else:
                reply_list = [
                    f"The sky is {coverage} {position}.",
                    f"{position.capitalize()} the sky is {coverage}.",
                ]

            return random.choice(reply_list)

        coverage_values = np.array(
            [CLOUD_COVER_LIMITS[coverage] for coverage in unique_categories]
        )
        coverage_values[unique_categories == "overcast"] = 100.0

        min_coverage = unique_categories[coverage_values.argmin()]
        max_coverage = unique_categories[coverage_values.argmax()]

        if position is None:
            reply_list = [
                f"The cloud coverage ranges from {min_coverage} to {max_coverage}.",
                f"During this trip the sky is {min_coverage} to {max_coverage}.",
                f"On this journey you will experience a {min_coverage} to {max_coverage} sky.",
            ]
        else:
            reply_list = [
                f"The cloud coverage ranges from {min_coverage} to {max_coverage} {position}.",
                f"{position.capitalize()} the sky is {min_coverage} to {max_coverage}.",
                f"On this journey you will experience a {min_coverage} to {max_coverage} sky {position}.",
            ]

        return random.choice(reply_list)

    def pre_process_df(self, route_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process the DataFrame to convert cloud cover values to descriptive text.

        Args:
            route_df (pd.DataFrame): DataFrame containing the route data with cloud cover values.

        Returns:
            pd.DataFrame: DataFrame with cloud cover values converted to descriptive text.

        """
        feature_names = route_df[self.FEATURE_NAME].apply(convert_cloud_cover_to_word)
        route_df["cloudCover_value"] = route_df[self.FEATURE_NAME]
        route_df[self.FEATURE_NAME] = feature_names
        return route_df
