import random

import numpy as np

from routellm.dataset.routes.const import (
    PROB_INTERVALS,
    PROBABILITIES,
    certainty_to_cat,
)
from routellm.dataset.verbalization.const_to_text import PROBABILITY_DESCRIPTION
from routellm.dataset.verbalization.tasks.categorical_base_task import (
    CategoricalBaseTask,
)


class IncidentCertaintyTask(CategoricalBaseTask):
    NAME = "feature-tasks.incident-certainty-task"
    VERSION = 0
    FEATURE_NAME = "incident_certainty"
    FILTER_LIST = ["rare"]
    CATEGORIES = PROBABILITIES

    def get_system_message(self, style_info=None):
        message, style = super().get_system_message(style_info)

        return (
            message
            + " This tasks aims at asking about the probability of traffic incidents. If not explicitly mentioned there are no incidents. The probabilities can be one of: certain, probable, risk of, improbable, rare.",
            style,
        )

    def _specific_generate_task(
        self, segments, target_value, has_multiple_segments, return_position, use_cot
    ):
        certainty_threshold = PROB_INTERVALS[target_value][0]

        has_feature = (segments["certainty_value"] >= certainty_threshold).any()
        if not has_multiple_segments or not has_feature:
            return (
                self._get_specific_response_true_single(
                    segments,
                    target_value,
                    use_cot,
                )
                if has_feature
                else self._get_specific_response_false(segments, target_value, use_cot)
            )

        # At least one closure
        travel_time_lengths = (
            segments.groupby("segment")["current_travel_time"].sum().cumsum()
        )

        # Define position as start = 1/3, middle = 2/3 and end = 3/3
        route_start_mid = 1 / 3 * segments["current_travel_time"].sum()
        route_mid_end = 2 / 3 * segments["current_travel_time"].sum()

        target_indices = np.where(
            segments[segments["start_segment"]]["certainty_value"]
            >= certainty_threshold
        )[0]

        plural = True

        # Check if only one segment has the feature
        if len(target_indices) == 1:
            plural = False

        journey_word = random.choice(
            [
                "of our ongoing trip",
                "of the trip",
                "of the route",
                "of the path",
                "of our path",
                "of your route",
                "of the journey",
                "of the provided route",
                "of your travel",
                "of this specific path",
                "of the way",
                "of the course",
                "of the itinerary",
            ]
        )
        position = "throughout the journey"

        min_feature_index = target_indices.min()
        max_feature_index = target_indices.max()
        if travel_time_lengths.iloc[max_feature_index] < route_start_mid:
            position = f"in the beginning {journey_word}"
        elif (
            route_start_mid
            <= travel_time_lengths.iloc[min_feature_index]
            <= travel_time_lengths.iloc[max_feature_index]
            < route_mid_end
        ):
            position = f"in the middle {journey_word}"
        elif route_mid_end <= travel_time_lengths.iloc[min_feature_index]:
            position = f"in the end {journey_word}"

        return self._get_specific_response_true_multi(
            segments,
            target_value,
            use_cot,
            position=position if return_position else None,
            plural=plural,
        )

    def _get_general_task(
        self,
    ):
        task_list = [
            "What are the various levels of probability for incidents occurring along this path",
            "How likely are we to encounter different types of incidents while traveling through this area",
            "What is the probability of encountering a traffic incident on this specific route",
            "Can you provide an estimated likelihood of experiencing an incident during this trip",
            "How likely are traffic incidents on this particular route",
            "Is there any data available to determine the certainty of facing an incident while traveling on this path",
            "What percentage chance exists for encountering a traffic incident on this journey",
            "Are traffic incidents highly probable or infrequent along this chosen route",
            "How frequently do incidents occur along this route",
            "How certain can we be about coming across a traffic incident during our travel on this road",
            "Can you offer insight into the likelihood of incidents occurring along our intended path",
            "Taking into account current conditions, how confident can we be that an incident might happen during our commute",
        ]

        return random.choice(task_list)

    def _get_specific_value_task(self, target_value):
        if target_value == "risk_of":
            task_list = [
                "Are there any with a risk of incidents on this trip",
                "Is there a risk of facing any incidents during our journey",
                "Does our route include the risk of any accidents or issues",
                "Is there a risk of experiencing traffic-related problems or obstacles",
                "Is there a risk of encountering some traffic events along our way",
                "When traveling on this route, are we risking encountering a traffic incident",
                "Do you foresee any instances with a risk of happening as part of our schedule",
            ]
        else:
            task_list = [
                f"Are there any {target_value} incidents on this trip",
                f"Will it be {target_value} to face incidents during our journey",
                f"Does our route include any {target_value} accidents or issues",
                f"Are we going to experience {target_value} problems or obstacles",
                f"Is there a {target_value} possibility that we will encounter traffic incidents or events along our way",
                f"Do we have any {target_value} occurrences happening in our travel plan",
                f"Do you foresee any instances with a {target_value} chance happening as part of our schedule",
            ]

        return random.choice(task_list)

    def _get_specific_response_true_single(self, segments, target_value, use_cot):
        cot_prepend_text = ""

        if use_cot:
            # get value range
            min_val = round(
                segments[segments[self.FEATURE_NAME] == target_value][
                    "certainty_value"
                ].min()
                * 100
            )
            max_val = round(
                segments[segments[self.FEATURE_NAME] == target_value][
                    "certainty_value"
                ].max()
                * 100
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

            description = random.choice(PROBABILITY_DESCRIPTION[target_value])

            cot_prepend_text = random.choice(
                [
                    f"The certainty of incidents occurring is {insert_text} for the whole route. Therefore the answer is: ",
                    f"Given the {insert_text} chance of an incident happening on this route, the correct answer is: ",
                    f"The question can be answered by inspecting the incident certainty on the route. In the provided route, the certainty is {insert_text}.\n",
                    f"The questions aims at understanding the incident certainty. The incident certainty describes the chance of an incident happening. It is either given in percent or in words ranging from 'rare' to 'certain'. On the route you provided the certainty is {insert_text}.\n\n",
                    f"With a {insert_text} chance the following classification is true:\n\n",
                    'To answer this question, it is also necessary to keep a definition of the incident certainty in mind:\n\nIncident certainty in traffic and travel refers to how likely it is for an event, like an accident or road closure, to occur while you\'re on the road. "Certain chance" means it\'s definitely going to happen, "probable chance" means there\'s a good likelihood of it happening, "risk of" means there\'s a possibility but not guaranteed, "improbable chance" means it\'s very unlikely to happen, and "rare chance" indicates that such an event is extremely uncommon.\n\n',
                    'You are asking about the probability of traffic incidents. In the context of traffic and travel, incident certainty denotes the likelihood of encountering an event such as a car crash or construction work during your journey. A "certain chance" implies that the occurrence is almost definite; a "probable chance" suggests that there is a high probability; "risk of" indicates that there\'s some potential but not necessarily high; an "improbable chance" conveys that the odds are low; and a "rare chance" signifies that such incidents seldom occur.\n\n',
                    "In order to derive the correct answer, the definition of incident certainty has to be consulted. Incident certainty within traffic management and transportation planning refers to quantifying the probability of events like accidents, congestion, or infrastructure disruptions impacting travelers' journeys. The various levels include:\n- Certain Chance: A virtually guaranteed incident.\n- Probable Chance: An event with considerable likelihood based on data analysis.\n- Risk Of: A potential occurrence with lower-than-average probability.\n- Improbable Chance: An event with minimal statistical evidence suggesting its likelihood.\n- Rare Chance: An extremely infrequent incident based on historical data or expert assessment.\n\n",
                    f"{description}\n\n",
                    f"Sure! {description}\n",
                    f"Understanding incident certainties helps with answering the question. {description}\n",
                ]
            )

        if target_value == "risk_of":
            reply_list = [
                "Yes, there are a few areas along this trip where you might encounter incidents with a risk of happening. Please be cautious and stay alert while driving.",
                "During your journey, there is the possibility of facing some incidents that have a risk of occurring. Make sure to pay attention to traffic conditions and any potential obstacles.",
                "Your route does include the risk of encountering some accidents or issues with a likelihood of taking place. Keep an eye out for any developing situations on the road.",
                "There is indeed a risk of experiencing traffic-related problems or obstacles during your travel, particularly those with a chance of happening. Staying vigilant will help ensure your safety.",
                "It appears that along your way, you may come across traffic events with a risk of occurring. Be prepared and consider alternative routes if needed.",
                "When traveling on this route, you do run the risk of encountering a traffic incident that has some likelihood of happening. It's essential to remain cautious and be ready for any unexpected situations.",
                "Based on current information, we foresee some instances with a risk of occurring during your scheduled trip. Drive carefully and pay close attention to your surroundings.",
                "There is the potential for coming across particular incidents with a risk of taking place while navigating through this route; thus, it's crucial to maintain awareness at all times.",
                "On this specific path, our analysis indicates that encountering events with a risk of occurring is possible - don't hesitate to adjust your plans accordingly if necessary.",
                "As per our data, there may be certain occurrences throughout this journey that have a risk of happening - make sure you stay informed about changing conditions while en route.",
            ]
        else:
            convert_to_adverb = {
                "certain": "certainly",
                "probable": "probably",
                "improbable": "improbably",
            }

            adverb_target = convert_to_adverb[target_value]
            reply_list = [
                f"On this trip, we can expect {target_value} incidents.",
                f"Our journey has a {target_value} chance of facing incidents.",
                f"During our route, we will {adverb_target} encounter accidents or issues.",
                f"There is a {target_value} possibility that we will experience problems or obstacles on the way.",
                f"The likelihood of encountering traffic incidents or events along our way is {target_value}.",
                f"In our travel plan, there are some occurrences with a {target_value} chance happening.",
                f"It is {target_value} that instances occur during our schedule.",
                f"Our route has the potential for {target_value} disturbances or complications.",
                f"We could come across traffic situations with a {target_value} probability during our journey.",
                f"Throughout our trip, there's a {target_value} chance of encountering incidents.",
            ]

        return cot_prepend_text + random.choice(reply_list)

    def _get_specific_response_true_multi(
        self, segments, target_value, use_cot, position=None, plural=False
    ):
        cot_prepend_text = ""

        if use_cot:
            # get value range
            certainty_threshold = PROB_INTERVALS[target_value][0]

            min_val = round(
                segments[segments["certainty_value"] >= certainty_threshold][
                    "certainty_value"
                ].min()
                * 100
            )
            max_val = round(
                segments[segments["certainty_value"] >= certainty_threshold][
                    "certainty_value"
                ].max()
                * 100
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

            description = random.choice(PROBABILITY_DESCRIPTION[target_value])

            cot_prepend_text = random.choice(
                [
                    f"The certainty of incidents occurring is {insert_text} {position if position is not None else 'in some segments of the route'}. Therefore the answer is: ",
                    f"Given the {insert_text} chance of an incident happening in sections of this route, the correct answer is: ",
                    f"The question can be answered by inspecting the incident certainty on the route. {position.capitalize() if position is not None else 'In some passages of the provided route'}, the certainty is {insert_text}.\n",
                    f"The questions aims at understanding the incident certainty. The incident certainty describes the chance of an incident happening. It is either given in percent or in words ranging from 'rare' to 'certain'. {'On the route you provided' if position is None else f'{position.capitalize()}'} the certainty is {insert_text}.\n\n",
                    f"With a {insert_text} chance the following classification is true:\n\n",
                    'To answer this question, it is also necessary to keep a definition of the incident certainty in mind:\n\nIncident certainty in traffic and travel refers to how likely it is for an event, like an accident or road closure, to occur while you\'re on the road. "Certain chance" means it\'s definitely going to happen, "probable chance" means there\'s a good likelihood of it happening, "risk of" means there\'s a possibility but not guaranteed, "improbable chance" means it\'s very unlikely to happen, and "rare chance" indicates that such an event is extremely uncommon.\n\n',
                    f'You are asking about the probability of traffic incidents. In the context of traffic and travel, incident certainty denotes the likelihood of encountering an event such as a car crash or construction work during your journey. A "certain chance" implies that the occurrence is almost definite; a "probable chance" suggests that there is a high probability; "risk of" indicates that there\'s some potential but not necessarily high; an "improbable chance" conveys that the odds are low; and a "rare chance" signifies that such incidents seldom occur.\n\\Keeping the {insert_text} incident certainty in mind, the correct answer is: ',
                    "In order to derive the correct answer, the definition of incident certainty has to be consulted. Incident certainty within traffic management and transportation planning refers to quantifying the probability of events like accidents, congestion, or infrastructure disruptions impacting travelers' journeys. The various levels include:\n- Certain Chance: A virtually guaranteed incident.\n- Probable Chance: An event with considerable likelihood based on data analysis.\n- Risk Of: A potential occurrence with lower-than-average probability.\n- Improbable Chance: An event with minimal statistical evidence suggesting its likelihood.\n- Rare Chance: An extremely infrequent incident based on historical data or expert assessment.\n\n",
                    f"{description}\n\n",
                    f"Sure! {description}\n",
                    f"Understanding incident certainties helps with answering the question. {description}\nGiven the {insert_text} incident certainty, the answer is: ",
                ]
            )

        if position is not None:
            if target_value == "risk_of":
                reply_list = [
                    f"Yes, {position} there are a few areas where you might encounter incidents with a risk of happening. Please be cautious and stay alert while driving.",
                    f"During your journey, particularly {position}, there is the possibility of facing some incidents that have a risk of occurring. Make sure to pay attention to traffic conditions and any potential obstacles.",
                    f"{position.capitalize()}, you may come across accidents or issues with a risk of taking place. Keep an eye out for any developing situations on the road.",
                    f"There is indeed a risk of experiencing traffic-related problems or obstacles {position}. Staying vigilant will help ensure your safety.",
                    f"It appears that {position}, you may come across traffic events with a risk of occurring. Be prepared and consider alternative routes if needed.",
                    f"When traveling {position} of this route, you do run the risk of encountering a traffic incident that has a risk of happening. It's essential to remain cautious and be ready for any unexpected situations.",
                    f"Based on current information ({position}), we foresee some instances with a risk of occurring during your scheduled trip. Drive carefully and pay close attention to your surroundings.",
                    f"There is potential for coming across particular incidents with a risk of taking place while navigating through this route ({position}); thus, it's crucial to maintain awareness at all times.",
                    f"{position.capitalize()}, our analysis indicates that encountering events with a risk of occurring is possible - don't hesitate to adjust your plans accordingly if necessary.",
                    f"As per our data, {position}, there may be certain occurrences throughout this journey that have a risk-of-happening - make sure you stay informed about changing conditions while en route.",
                ]
            else:
                convert_to_adverb = {
                    "certain": "certainly",
                    "probable": "probably",
                    "improbable": "improbably",
                }

                adverb_target = convert_to_adverb[target_value]

                reply_list = [
                    f"{target_value.capitalize()} incidents are expected {position}.",
                    f"{position.capitalize()}, our journey has a {target_value} chance of facing incidents.",
                    f"Yes, we will {adverb_target} encounter accidents or issues {position}.",
                    f"There is {target_value} possibility that we will experience problems or obstacles {position}.",
                    f"{position.capitalize()}, we can {adverb_target} expect incidents.",
                    f"{position.capitalize()}, our journey has a {target_value} chance of facing incidents.",
                    f"During our route, {position}, we will {adverb_target} encounter accidents or issues.",
                    f"There is a {target_value} possibility that we will experience problems or obstacles {position}.",
                    f"{position.capitalize()}, the likelihood of encountering traffic incidents or events is {target_value}.",
                    f"During the travel plan, {position}, there are some occurrences with a {target_value} chance of happening.",
                    f"It is more likely for instances to occur at a(n) {target_value} level when traveling {position}.",
                ]

        else:
            if target_value == "risk_of":
                reply_list = [
                    "Yes, there are a few areas along this trip where you might encounter incidents with a risk of happening. Please be cautious and stay alert while driving.",
                    "During your journey, there is the possibility of facing some incidents that have a risk of occurring. Make sure to pay attention to traffic conditions and any potential obstacles.",
                    "Your route does include the risk of encountering some accidents or issues with a likelihood of taking place. Keep an eye out for any developing situations on the road.",
                    "There is indeed a risk of experiencing traffic-related problems or obstacles during your travel, particularly those with a chance of happening. Staying vigilant will help ensure your safety.",
                    "It appears that along your way, you may come across traffic events with a risk of occurring. Be prepared and consider alternative routes if needed.",
                    "When traveling on this route, you do run the risk of encountering a traffic incident that has some likelihood of happening. It's essential to remain cautious and be ready for any unexpected situations.",
                    "Based on current information, we foresee some instances with a risk of occurring during your scheduled trip. Drive carefully and pay close attention to your surroundings.",
                    "There is the potential for coming across particular incidents with a risk of taking place while navigating through this route; thus, it's crucial to maintain awareness at all times.",
                    "On this specific path, our analysis indicates that encountering events with a risk of occurring is possible - don't hesitate to adjust your plans accordingly if necessary.",
                    "As per our data, there may be certain occurrences throughout this journey that have a risk of happening - make sure you stay informed about changing conditions while en route.",
                ]
            else:
                convert_to_adverb = {
                    "certain": "certainly",
                    "probable": "probably",
                    "improbable": "improbably",
                }

                adverb_target = convert_to_adverb[target_value]
                reply_list = [
                    f"On this trip, we can expect {target_value} incidents.",
                    f"Our journey has a {target_value} chance of facing incidents.",
                    f"During our route, we will encounter {adverb_target} accidents or issues.",
                    f"Yes, here is a {target_value} possibility that we will experience problems or obstacles on the way.",
                    f"The likelihood of encountering traffic incidents or events along our way is {target_value}.",
                    f"In our travel plan, there are some occurrences with a {target_value} chance happening.",
                    f"It is {target_value} that instances occur during our schedule.",
                    f"Indeed, our route has the potential for {target_value} disturbances or complications.",
                    f"We could come across traffic situations with a {target_value} probability during our journey.",
                    f"Throughout our trip, there's a {target_value} chance of encountering incidents.",
                ]

        return cot_prepend_text + random.choice(reply_list)

    def _get_specific_response_false(self, segments, target_value, use_cot):
        cot_prepend_text = ""

        if use_cot:
            # get value range
            min_val = round(segments["certainty_value"].min() * 100)
            max_val = round(segments["certainty_value"].max() * 100)

            if min_val == max_val:
                insert_text = random.choice([f"{min_val} %", f"{min_val} percent"])
            else:
                insert_text = random.choice(
                    [
                        f"{min_val} to {max_val} %",
                        f"{min_val} to {max_val} percent",
                    ]
                )

            description = random.choice(PROBABILITY_DESCRIPTION[target_value])

            cot_prepend_text = random.choice(
                [
                    f"The certainty of incidents occurring is {insert_text} for the whole route. Therefore the answer is: ",
                    f"Given the {insert_text} chance of an incident happening in sections of this route, the correct answer is: ",
                    f"The question can be answered by inspecting the incident certainty on the route. For the entirety of the provided route, the certainty is {insert_text}.\n",
                    f"The questions aims at understanding the incident certainty. The incident certainty describes the chance of an incident happening. It is either given in percent or in words ranging from 'rare' to 'certain'. On the route you provided the certainty is {insert_text}.\n\n",
                    f"With a {insert_text} chance the following classification is true:\n\n",
                    'To answer this question, it is also necessary to keep a definition of the incident certainty in mind:\n\nIncident certainty in traffic and travel refers to how likely it is for an event, like an accident or road closure, to occur while you\'re on the road. "Certain chance" means it\'s definitely going to happen, "probable chance" means there\'s a good likelihood of it happening, "risk of" means there\'s a possibility but not guaranteed, "improbable chance" means it\'s very unlikely to happen, and "rare chance" indicates that such an event is extremely uncommon.\n\n',
                    f'You are asking about the probability of traffic incidents. In the context of traffic and travel, incident certainty denotes the likelihood of encountering an event such as a car crash or construction work during your journey. A "certain chance" implies that the occurrence is almost definite; a "probable chance" suggests that there is a high probability; "risk of" indicates that there\'s some potential but not necessarily high; an "improbable chance" conveys that the odds are low; and a "rare chance" signifies that such incidents seldom occur.\n\\Keeping the {insert_text} incident certainty in mind, the correct answer is: ',
                    "In order to derive the correct answer, the definition of incident certainty has to be consulted. Incident certainty within traffic management and transportation planning refers to quantifying the probability of events like accidents, congestion, or infrastructure disruptions impacting travelers' journeys. The various levels include:\n- Certain Chance: A virtually guaranteed incident.\n- Probable Chance: An event with considerable likelihood based on data analysis.\n- Risk Of: A potential occurrence with lower-than-average probability.\n- Improbable Chance: An event with minimal statistical evidence suggesting its likelihood.\n- Rare Chance: An extremely infrequent incident based on historical data or expert assessment.\n\n",
                    f"{description}\n\n",
                    f"Sure! {description}\n",
                    f"Understanding incident certainties helps with answering the question. {description}\nGiven the {insert_text} incident certainty, the answer is: ",
                ]
            )

        if target_value == "risk_of":
            reply_list = [
                "There is no risk of incidents on this trip.",
                "There is no risk of facing any incidents during our journey.",
                "Our route does not include the risk of any accidents or issues.",
                "There is no risk of experiencing traffic-related problems or obstacles.",
                "There is no risk of encountering some traffic events along our way.",
                "When traveling on this route, we are not risking encountering a traffic incident.",
                "I do not foresee any instances with a risk of happening as part of our schedule.",
                "All incidents on this route are improbable.",
            ]
        else:
            reply_list = [
                f"There are no {target_value} incidents on this trip.",
                f"It is not {target_value} that we will face incidents during our journey.",
                f"Our route does not include any {target_value} accidents or issues.",
                f"We are not going to experience {target_value} problems or obstacles.",
                f"We have no {target_value} occurrences happening in our travel plan.",
                f"I do not foresee any instances with a {target_value} chance happening as part of our schedule.",
            ]

        return cot_prepend_text + random.choice(reply_list)

    def _categories_to_text(self, unique_categories, position=None):
        u_cat = [cert for cert in unique_categories if "rare" not in cert]

        if len(u_cat) == 0:
            # No incidents
            reply_list = [
                "No incidents were found on this route.",
                "I did not detect any incident of significant certainty on our path.",
                "There are no reported incidents on this journey.",
                "It looks like you won't encounter any incidents during your travel.",
                "Our trip appears to be clear of any incidents or disruptions.",
                "Good news! There's no indication of any incident of significant probability along our course.",
            ]

            return random.choice(reply_list)

        if len(u_cat) == 1:
            # only one incident magnitude
            certainty_value = u_cat[0]

            if certainty_value == "risk_of":
                if position is not None:
                    reply_list = [
                        f"{position.capitalize()} there are a few areas where you might encounter incidents with a risk of happening.",
                        f"{position.capitalize()}, there is the possibility of facing some incidents that have a risk of occurring.",
                        f"{position.capitalize()}, you may come across accidents or issues with a risk of taking place.",
                        f"There is a risk of experiencing traffic-related problems or obstacles {position}.",
                        f"It appears that {position}, you may come across traffic events with a risk of occurring.",
                    ]

                else:
                    reply_list = [
                        "On this route, there are a few areas where you might encounter incidents with a risk of happening.",
                        "While driving on this route, there is the possibility of facing some incidents that have a risk of occurring.",
                        "On the while trip you may come across accidents or issues with a risk of taking place.",
                        "There is a risk of experiencing traffic-related problems or obstacles.",
                        "It appears that during this journey, you may come across traffic events with a risk of occurring.",
                    ]
            else:
                convert_to_adverb = {
                    "certain": "certainly",
                    "probable": "probably",
                    "improbable": "improbably",
                }

                adverb_target = convert_to_adverb[certainty_value]

                if position is None:
                    reply_list = [
                        f"On this trip, we can expect {certainty_value} incidents.",
                        f"The journey has a {certainty_value} chance of facing incidents.",
                        f"During this route, we will encounter {adverb_target} accidents or issues.",
                        f"There is a {certainty_value} possibility that we will experience problems or obstacles on the way.",
                        f"The likelihood of encountering traffic incidents or events along our way is {certainty_value}.",
                        f"In our travel plan, there are some occurrences with a {certainty_value} chance happening.",
                        f"It is {certainty_value} that instances occur during our schedule.",
                        f"Our route has the potential for {certainty_value} disturbances or complications.",
                        f"We could come across traffic situations with a {certainty_value} probability during our journey.",
                        f"Throughout our trip, there's a {certainty_value} chance of encountering incidents.",
                    ]
                else:
                    reply_list = [
                        f"{certainty_value.capitalize()} incidents are expected {position}.",
                        f"{position.capitalize()}, our journey has a {certainty_value} chance of facing incidents.",
                        f"We will {adverb_target} encounter accidents or issues {position}.",
                        f"There is {certainty_value} possibility that we will experience problems or obstacles {position}.",
                        f"{position.capitalize()}, we can {adverb_target} expect incidents.",
                        f"{position.capitalize()}, our journey has a {certainty_value} chance of facing incidents.",
                        f"During our route, {position}, we will {adverb_target} encounter accidents or issues.",
                        f"There is a {certainty_value} possibility that we will experience problems or obstacles {position}.",
                        f"{position.capitalize()}, the likelihood of encountering traffic incidents or events is {certainty_value}.",
                        f"During the travel plan, {position}, there are some occurrences with a {certainty_value} chance of happening.",
                        f"It is more likely for instances to occur at a(n) {certainty_value} level when traveling {position}.",
                    ]

            return random.choice(reply_list)

        certainty_values = np.array([PROBABILITIES.index(cert) for cert in u_cat])

        min_certainty_word = u_cat[certainty_values.argmax()]
        max_certainty_word = u_cat[certainty_values.argmin()]

        if min_certainty_word == "risk_of":
            reply_list = [
                f"The incidents {'on this journey' if position is None else position} all have at least a risk of happening. The certainty ranges up to {max_certainty_word} incidents.",
                f"There is at least a risk of several incidents occurring. The most probable incidents {'on this route' if position is None else position} have a {max_certainty_word} chance of arising.",
                f"Each incident {'in this route' if position is None else position} has a risk of occurring. The incidents with the highest chance of happening are {max_certainty_word}.",
            ]
        elif max_certainty_word == "risk_of":
            reply_list = [
                f"Incidents with varying probability starting from {min_certainty_word} incidents to incidents with a risk of occurring can be expected {'throughout various segments of our chosen course' if position is None else position}.",
                f"Stretches {'of this journey' if position is None else position} contain events with a {min_certainty_word} certainty up to incidents with a risk of happening.",
                f"{'While traversing this planned trip' if position is None else position.capitalize()}, we should anticipate encountering a variety of incidents some of which are {min_certainty_word} while others have a risk of taking place.",
            ]
        else:
            reply_list = [
                f"Incidents with a probability ranging from {min_certainty_word} to {max_certainty_word} can be expected {'throughout various segments of our chosen course' if position is None else position}.",
                f"Stretches {'of this journey' if position is None else position} contain events with a certainty rated anywhere between {min_certainty_word} and {max_certainty_word}.",
                f"{'While traversing this planned trip' if position is None else position.capitalize()}, we should anticipate encountering a variety of incidents with probabilities spanning from {min_certainty_word} all the way up to {max_certainty_word}.",
                f"{'Throughout our journey' if position is None else position.capitalize()}, incident probabilities will vary widely—ranging from {min_certainty_word} incidents to {max_certainty_word} ones.",
            ]

        return random.choice(reply_list)

    def pre_process_df(self, route_df):
        feature_names = route_df[self.FEATURE_NAME].apply(certainty_to_cat)
        route_df["certainty_value"] = route_df[self.FEATURE_NAME]
        route_df[self.FEATURE_NAME] = feature_names
        return route_df
