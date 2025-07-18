import random

import numpy as np

from routellm.dataset.routes.const import MAGNITUDES
from routellm.dataset.verbalization.const_to_text import MAGNITUDE_DESCRIPTIONS
from routellm.dataset.verbalization.tasks.categorical_base_task import (
    CategoricalBaseTask,
)


class IncidentMagnitudeTask(CategoricalBaseTask):
    NAME = "feature-tasks.incident-magnitude-task"
    VERSION = 0
    FEATURE_NAME = "incident_magnitude"
    CATEGORIES = [mag.lower() for mag in MAGNITUDES if "unknown" not in mag.lower()]

    def _get_general_task(
        self,
    ):
        task_list = [
            "What incident magnitudes are encountered on this route",
            "Can you list the magnitudes of incidents present along the way",
            "Which severity levels of incidents can be found on our route",
            "Are there any specific magnitude levels for incidents we should be aware of while traveling",
            "What are the different severity levels of incidents on this path",
            "Please provide the incident magnitude classifications for this journey",
            "Tell me about any incident magnitude categories that we may come across during our travel",
            "Describe the range of incident magnitudes we might face en route",
            "How would you classify the magnitudes of incidents occurring on this route",
            "Identify any notable incident magnitude levels for our trip",
            "Could you inform me about various degrees of incident severities present in our path",
            "I'd like to know about any particular magnitudes associated with incidents along this route. Can you help with that",
        ]

        return random.choice(task_list)

    def _get_specific_value_task(self, target_value):
        task_list = [
            f"Are there any {target_value} incidents on this trip",
            f"Can you find any {target_value} magnitude incidents along our route",
            f"Do we have to deal with any {target_value}-level incidents during our journey",
            f"Could you inform me about the presence of {target_value} magnitude incidents on this path",
            f"Please tell me if there are any {target_value}-severity incidents that we may encounter",
            f"I'd like to know if we'll come across any incidents classified as {target_value}",
            f"Will we face any {target_value}-magnitude incidents en route",
            f"How likely are we to encounter a {target_value} incident during our travel",
            f"What are the chances of coming across a {target_value}-level incident on this route",
            f"Do we need to be prepared for any specific {target_value} severity events while traveling",
        ]

        return random.choice(task_list)

    def _get_specific_response_true_single(self, segments, target_value, use_cot):
        description_string = random.choice(MAGNITUDE_DESCRIPTIONS[target_value])

        cot_prepend_text = ""

        if use_cot:
            cot_prepend_text = random.choice(
                [
                    f"This is based on the incident impact being described as: {description_string}. You can expect a {target_value} impact on your route. Please note that this information provides an overview of the potential disruptions you may experience during your journey. By referring to the provided description, you can better anticipate what to expect and make informed decisions regarding your travel plans.\n\nThis results in my answer: ",
                    f"In case of a {target_value} impact, the following description holds: {description_string}.\n\nGiven this description the following is the correct answer: ",
                    f"The question asks about {target_value} impact incidents on the route. {description_string}.\n\nTherefore: ",
                    f"An incident of {target_value} impact is described by:\n\n\t{description_string}.\nThe description can be used to analyze the provided journey.\n\n",
                    f"This can be answered by considering the possibility of a {target_value} incident and its effects:\n\n{description_string}.\n\nResult: ",
                    f"{target_value.capitalize()} impacts can be described with the following text:\n\n{description_string}.\nThe answer when comparing the description with the route at hand is: ",
                    "From the perspective of drivers and passengers, incident magnitude in traffic signifies the severity of a traffic issue they encounter, such as an accident or roadblock. Understanding its scale enables them to make informed decisions about their journey and potential adjustments.\n\nThe answer based on this description: ",
                    "For drivers and passengers, assessing traffic incident magnitude involves evaluating the perceived severity of an encountered disruption (e.g., collisions or obstructions). This perception informs decision-making regarding route alterations, safety precautions, and overall situational awareness during transportation.\n\n",
                    f"The following tokens indicate a {target_value} impact: {self.TOKEN_REASON_REPLACEMENT_TEXT}.\n\n",
                    f"Indicators for a {target_value} impact are: {self.TOKEN_REASON_REPLACEMENT_TEXT}.\n\n",
                    f"These tokens signify a {target_value} impact: {self.TOKEN_REASON_REPLACEMENT_TEXT}.\n\n",
                ]
            )

        reply_list = [
            f"My findings show that a {target_value}-caliber event will indeed transpire along our planned course. Be sure to make necessary preparations!",
            f"My thorough examination has proven fruitful: You may anticipate coming into contact with an incident ranked as {target_value} while traversing this particular path.",
            f"The results are conclusive—our research indicates that {target_value} level situations aren't just probable but guaranteed as we navigate through our chosen direction.",
            f"My assessment reveals that we'll indeed face {target_value} severity events throughout various stretches of our ongoing trip.",
            f"Yes, there is at least one {target_value} incident on this route.",
            f"I have found instances of {target_value}-magnitude incidents along the way.",
            f"Indeed, we can expect to encounter a {target_value} incident during our journey.",
            f"Upon checking, I discovered that there are some {target_value} severity incidents on this route.",
            f"It looks like you might come across a(n) {target_value}-level incident while traveling.",
            f"After analyzing the data, it appears that there are {target_value} magnitude events en route.",
            f"Certainly! I can confirm the presence of a {target_value}-severity incident in our path.",
            f"To address your concern, yes, we do have at least one {target_value}-magnitude event occurring along this route. So stay prepared!",
            f"Based on my analysis of the available information, it seems that encountering a {target_value} incident is possible during our travel. Keep an eye out for any relevant updates or changes!",
        ]

        return cot_prepend_text + random.choice(reply_list)

    def _get_specific_response_true_multi(
        self, segments, target_value, use_cot, position=None, plural=False
    ):
        description_string = random.choice(MAGNITUDE_DESCRIPTIONS[target_value])

        cot_prepend_text = ""

        if use_cot:
            cot_prepend_text = random.choice(
                [
                    f"This is based on the incident impact being described as: {description_string}. You can expect a {target_value} impact on your route. Please note that this information provides an overview of the potential disruptions you may experience during your journey. By referring to the provided description, you can better anticipate what to expect and make informed decisions regarding your travel plans.\n\nThis results in my answer: ",
                    f"In case of a {target_value} impact, the following description holds: {description_string}.\n\nGiven this description the following is the correct answer: ",
                    f"The question asks about {target_value} impact incidents on the route. {description_string}.\n\nTherefore: ",
                    f"An incident of {target_value} impact is described by:\n\n\t{description_string}.\nThe description can be used to analyze the provided journey.\n\n",
                    f"This can be answered by considering the possibility of a {target_value} incident and its effects:\n\n{description_string}.\n\nResult: ",
                    f"{target_value.capitalize()} impacts can be described with the following text:\n\n{description_string}.\nThe answer when comparing the description with the route at hand is: ",
                    "From the perspective of drivers and passengers, incident magnitude in traffic signifies the severity of a traffic issue they encounter, such as an accident or roadblock. Understanding its scale enables them to make informed decisions about their journey and potential adjustments.\n\nThe answer based on this description: ",
                    "For drivers and passengers, assessing traffic incident magnitude involves evaluating the perceived severity of an encountered disruption (e.g., collisions or obstructions). This perception informs decision-making regarding route alterations, safety precautions, and overall situational awareness during transportation.\n\n",
                    f"The following tokens indicate a {target_value} impact: {self.TOKEN_REASON_REPLACEMENT_TEXT}.\n\n",
                    f"Indicators for a {target_value} impact are: {self.TOKEN_REASON_REPLACEMENT_TEXT}.\n\n",
                    f"These tokens signify a {target_value} impact: {self.TOKEN_REASON_REPLACEMENT_TEXT}.\n\n",
                ]
            )

        if position is not None:
            reply_list = [
                f"Yes, there are incidents of {target_value} impact {position}.",
                f"I've found that {target_value}-magnitude incidents occur {position}.",
                f"Indeed, we can expect to encounter a {target_value} incident {position}.",
                f"Our analysis indicates that {target_value}-severity incidents are located {position}.",
                f"It seems like you might come across multiple occurrences of {target_value}-level incidents situated {position}.",
                f"In case you're curious about {target_value}-caliber events: They will indeed transpire along our planned course and are situated {position}.",
                f"My research has produced exciting results: We'll be encountering {target_value} severity events throughout various stretches {position}.",
                f"A close examination reveals that numerous situations ranked as {target_value} impact arise {position}.",
            ]
        else:
            reply_list = [
                f"Yes, there are instances of incidents with {target_value} impact on several parts of the route.",
                f"There seem to be multiple occurrences involving incidents with a {target_value} severity level, scattered throughout various stretches.",
                f"While examining available data concerning your query about {target_value} impact incidents, I've concluded that such situations aren't limited to one area but will likely pop up all along our planned voyage.",
                f"Yes there are sections with {target_value} incidents.",
                f"There are passages of the route with {target_value} incidents.",
                f"I've found that {target_value}-magnitude incidents occur in our path.",
                f"It seems like you might come across multiple occurrences of {target_value}-level incidents.",
                f"In case you're curious about {target_value}-caliber events: They will indeed transpire along our planned course.",
                f"We'll be encountering {target_value} severity incidents throughout various stretches of our ongoing trip.",
                f'A close examination reveals that numerous situations ranked as "{target_value} impact" incidents are scheduled to arise during this route.',
            ]

        return cot_prepend_text + random.choice(reply_list)

    def _get_specific_response_false(self, segments, target_value, use_cot):
        reply_list = [
            f"No, there are no instances of {target_value} incidents anywhere in the route.",
            f"I could not find any evidence of a(n) {target_value}-level incident on this path.",
            f"Don't worry; my search results show no signs of any {target_value}-severity events during our journey.",
            f"It appears that we will not face any incidents classified as '{target_value}'-magnitude along this route.",
            f"There seem to be no reported occurrences involving incidents with a {target_value} severity level.",
            f'Rest assured: My investigation shows zero instances where we\'d be confronted by an occurrence ranked as "{target_value}" throughout our entire trip.',
            f"Based on the given route, we don't need to be concerned about coming into contact with {target_value} incidents — there don't appear to be any on this specific course.",
            f"While examining available data concerning your query about incidents of {target_value} magnitude, I've concluded that such situations aren't part-and-parcel with our planned voyage.",
            f"No, there are no instances of {target_value} incidents anywhere in the route.",
            f"I could not find any evidence of {target_value}-level incidents on this path.",
            f"Don't worry; my search results show no signs of any {target_value}-severity events during our journey.",
            f"It appears that we will not face any incidents classified as '{target_value}'-magnitude along this route.",
        ]

        response = random.choice(reply_list)

        if use_cot:
            response += " " + self._categories_to_text(
                segments[self.FEATURE_NAME].unique().tolist()
            )

        return response

    def _categories_to_text(self, unique_categories, position=None):
        u_cat = [mag for mag in unique_categories if "unknown" not in mag]

        if len(u_cat) == 0:
            # No incidents
            reply_list = [
                "No incidents were found on this route.",
                "I did not detect any incident of significant magnitude on our path.",
                "There are no reported incidents on this journey.",
                "It looks like you won't encounter any incidents during your travel.",
                "Our trip appears to be clear of any incidents or disruptions.",
                "Good news! There's no indication of any incident of significant impact along our course.",
            ]

            return random.choice(reply_list)

        if len(u_cat) == 1:
            # only one incident magnitude
            magnitude_value = u_cat[0]
            if position is None:
                reply_list = [
                    f"The only incident encountered on this route is of {magnitude_value} impact.",
                    f"There's just one type of severity to note: {magnitude_value} incidents along our path.",
                    f"We can expect to find {magnitude_value}-level incidents during our journey.",
                    f"My analysis indicates that we'll primarily confront traffic incidents classified as '{magnitude_value}'.",
                    f'All identified traffic incidents share a common classification; they fall under the "{magnitude_value}" category.',
                ]
            else:
                reply_list = [
                    f"{position.capitalize()}, we will encounter {magnitude_value}-severity incidents during our journey.",
                    f"{position.capitalize()}, there are instances of {magnitude_value} magnitude events.",
                    f"Our voyage includes a stretch—situated {position}—during which we'll be confronted by traffic incidents labeled as {magnitude_value} impact.",
                    f"{position.capitalize()} of our journey, we can expect to encounter {magnitude_value}-severity incidents.",
                    f"Incidents classified as '{magnitude_value}' are present {position}.",
                ]

            return random.choice(reply_list)

        magnitude_values = np.array(
            [MAGNITUDES.index(mag.capitalize()) for mag in u_cat]
        )

        min_magnitude_word = u_cat[magnitude_values.argmin()]
        max_magnitude_word = u_cat[magnitude_values.argmax()]

        if position is None:
            # Include position and describe the magnitude range on this trip
            reply_list = [
                f"Incidents ranging from {min_magnitude_word} to {max_magnitude_word} severity can be expected throughout various segments of our chosen course.",
                f"Our investigation shows that different stretches of this journey contain events rated anywhere between {min_magnitude_word} and {max_magnitude_word}.",
                f"While traversing this planned trip, we should anticipate encountering a variety of incidents spanning from {min_magnitude_word} all the way up to {max_magnitude_word}.",
                f"Throughout our journey, incident magnitudes will vary widely—ranging from {min_magnitude_word}-caliber incidents to those labeled as {max_magnitude_word}.",
                f"A myriad of situations awaits us during our voyage; these could include anything from {min_magnitude_word} up until events graded as {max_magnitude_word} impact.",
            ]
        else:
            # Describe the magnitude range on this trip
            reply_list = [
                f"{position.capitalize()}, we can expect incidents ranging from {min_magnitude_word} to {max_magnitude_word} severity.",
                f"During our journey, {position} will have incidents spanning magnitudes from {min_magnitude_word} to {max_magnitude_word}.",
                f"In our path, {position}, we'll come across a variety of situations—ranging in severity from {min_magnitude_word} up to {max_magnitude_word}.",
                f"The stretch located {position} segment features event severities varying between {min_magnitude_word} and {max_magnitude_word} classifications.",
                f"At the specified area {position}, multiple types of incidents with differing intensities—from {min_magnitude_word} through {max_magnitude_word}—are known for springing into action.",
            ]

        return random.choice(reply_list)

    def pre_process_df(self, route_df):
        feature_names = route_df[self.FEATURE_NAME].apply(
            lambda x: MAGNITUDES[x].lower()
        )
        route_df["magnitudes_value"] = route_df[self.FEATURE_NAME]
        route_df[self.FEATURE_NAME] = feature_names
        return route_df
