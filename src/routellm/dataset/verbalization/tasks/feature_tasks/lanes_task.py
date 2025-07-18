import random

from routellm.dataset.verbalization.tasks.categorical_base_task import (
    CategoricalBaseTask,
)


class LanesTask(CategoricalBaseTask):
    NAME = "feature-tasks.lanes-task"
    VERSION = 0
    FEATURE_NAME = "lanes"
    CATEGORIES = list(range(1, 13))

    COT_CHANCE = 0.0  # NO COT

    def get_task(self, unique_values):
        punctuation_characters = [
            "? ",
            ". ",
            ": ",
            " ",
            "  ",
            "\n ",
            "\n\n ",
        ]

        task_type = random.choice(
            ["general", *["specific" for _ in range(len(self.CATEGORIES))]]
        )
        use_cot = random.random() < self.COT_CHANCE
        return_position = random.random() < self.POSITION_CHANCE

        if task_type == "general":
            task_text, target_value = self._get_general_task()
            self.task_mode = target_value
        else:
            target_value = max(
                1, round(random.gauss(3.0, 2.0))
            )  # Sample from normal distribution
            task_text = self._get_specific_value_task(target_value)

            if use_cot and return_position:
                task_text += (
                    random.choice(["?", ".", "!", "?\n", ".\n", "!\n", "\n", "\n\n"])
                    + " "
                    + random.choice(
                        [
                            "Please explain your decision and say where along the route",
                            "Why and when",
                            "Explain why and when",
                            "Add why and when",
                            "Please explain why and add the position along the route",
                            "Provide the reason why and mention when",
                        ]
                    )
                )
            elif use_cot:
                task_text += random.choice(
                    ["? ", ". ", "! ", "?\n", ".\n", "!\n", "\n", "\n\n"]
                ) + random.choice(
                    [
                        "Please explain your decision",
                        "Provide reasons for your answer",
                        "In addition to the answer, explain why it is correct",
                        "Explain your answer",
                        "Please explain",
                    ]
                )

            elif return_position:
                task_text += (
                    random.choice(["?", ".", "!", "?\n", ".\n", "!\n", "\n", "\n\n"])
                    + " "
                    + random.choice(
                        [
                            "Where along the route",
                            "Also answer where in the route",
                            "Please add where",
                            "Where",
                            "When",
                            "When along the route",
                        ]
                    )
                )

        return (
            task_text + random.choice(punctuation_characters),
            target_value,
            task_type,
            use_cot,
            return_position,
        )

    def _get_general_task(
        self,
    ):
        max_task_list = [
            *[
                f"What is the maximum amount of lanes on this {word}"
                for word in ["route", "journey", "trip"]
            ],
            "Can you provide me with the maximum number of lanes",
            "How high is the maximum number of lanes on this route",
            *[
                f"How many lanes are there at the widest point on this {word}"
                for word in ["route", "journey", "trip"]
            ],
            "What's the greatest number of lanes I will encounter during my drive",
            "During my travel, how many lanes will I see at most",
            "Please inform me about the highest number of lanes on this path",
            "At any point along this route, what's the maximum count of lanes",
        ]

        min_task_list = [
            *[
                f"What is the minimum amount of lanes on this {word}"
                for word in ["route", "journey", "trip"]
            ],
            "Can you provide me with the minimum number of lanes",
            "I will be driving the route at hand and I was wondering, what the minimum number of lanes is",
            *[
                f"How many lanes are there at the narrowest point on this {word}"
                for word in ["route", "journey", "trip"]
            ],
            "What's the least number of lanes I will encounter during my drive",
            "During my travel, how many lanes will I see at the minimum",
            "Please inform me about the lowest number of lanes on this path",
            "At any point along this route, what's the minimum count of lanes",
        ]

        task_type = random.choice(["min", "max"])

        text = random.choice(max_task_list if task_type == "max" else min_task_list)

        return text, task_type

    def _get_specific_value_task(self, target_value):
        task_list = [
            f"Does this route have any sections with {round(target_value)} lanes",
            f"Can you tell me whether this route has a portion where there are exactly {round(target_value)} lanes",
            f"While passing through this route, are there any instances, with {round(target_value)} lanes",
            f"Is there a street with {round(target_value)} lanes",
            f"Are there any {round(target_value)} lane streets",
            f"Is there a part of the trip where we will drive over a road where the number of lanes is {round(target_value)}",
            f"Will there be a {round(target_value)} lane street",
            f"Is it possible to find a section with {round(target_value)} lanes on this route",
            f"Does the route feature any stretches with {round(target_value)} lanes",
            f"Are there segments of the route that have {round(target_value)} lanes",
            f"Will we encounter any roads with {round(target_value)} lanes during our journey",
            f"Can I expect to see any {round(target_value)} lane roads along this route",
            f"Is there any part of the route where the number of lanes equals {round(target_value)}",
            f"Does this route include any roads with {round(target_value)} lanes",
            f"Do we come across any streets with {round(target_value)} lanes while traveling this route",
            f"Will we pass any sections with {round(target_value)} lanes on this route",
            f"Are we going to traverse any roads with {round(target_value)} lanes along the way",
            f"On this particular route, is there any specific segment or section where we might encounter a road that features {round(target_value)} lane",
            f"While traveling along this route, can you provide information on whether we will come across any stretches of road that possess {round(target_value)} lanes, which could be essential for understanding the route's infrastructure and planning our journey accordingly",
        ]

        return random.choice(task_list)

    def _get_specific_response_true_single(self, segments, target_value, use_cot):
        text_target = str(round(target_value))

        reply_list = [
            f"Yes, this route does have sections with the exact {text_target} lanes throughout the journey.",
            f"Indeed, this route features a portion where there are precisely {text_target} lanes.",
            f"As you travel this route, you will encounter instances with {text_target} lanes.",
            f"There is a street on this route with {text_target} lanes.",
            f"You will find {text_target} lane streets on this route.",
            f"During the trip, you will come across a road with {text_target} lanes.",
            f"Yes, you will encounter a {text_target} lane street on this journey.",
            f"It is possible to find a section with {text_target} lanes along this route.",
            f"The route features stretches with {text_target} lanes.",
            f"Segments of the route have {text_target} lanes.",
            f"You will encounter roads with {text_target} lanes during your journey.",
            f"Expect to see {text_target} lane roads along this route.",
            f"There is a part of the route where the number of lanes equals {text_target}.",
            f"This route includes roads with {text_target} lanes.",
            f"Streets with {text_target} lanes will be encountered while traveling this route.",
            f"Pass sections with {text_target} lanes on this route.",
            f"Traverse roads with {text_target} lanes along the way.",
            f"On this particular route, encounter a road with {text_target} lanes.",
            f"While traveling along this route, you will come across stretches of road that possess {text_target} lanes.",
            f"In this particular route, you will find that all sections maintain a consistent number of lanes. As such, you can expect to traverse roads with exactly {text_target} lanes throughout your journey, providing a uniform driving experience.",
            f"As you travel along this route, it's worth noting that there are no deviations in the number of lanes present. The entire journey is characterized by roads with exactly {text_target} lanes, ensuring a consistent and predictable driving environment.",
            f"Throughout your entire journey on this route, you'll notice that there is a remarkable consistency in the infrastructure. Every segment and section features roads with exactly {text_target} lanes, maintaining a uniform experience for drivers from start to finish.",
            f"During your journey on this specific route, you will find that the number of lanes remains constant, providing a harmonious driving experience. The route is characterized by roads with precisely {text_target} lanes, which are designated paths for vehicles to travel in the same direction. Lanes play a crucial role in organizing traffic, promoting safety, and allowing for efficient movement of vehicles. In this particular instance, the consistent presence of {text_target} lanes throughout the route ensures a uniform driving environment, aiding in navigation and traffic management. This consistency can be beneficial for drivers who appreciate predictability and well-structured roadways during their travels.",
        ]

        if use_cot:
            raise NotImplementedError("No template for chain-of-thought!")

        return random.choice(reply_list)

    def _get_specific_response_true_multi(
        self, segments, target_value, use_cot, position=None, plural=False
    ):
        text_target = str(round(target_value))

        if position is not None:
            reply_list = [
                f"Yes, this route does have sections with the exact {text_target} lanes {position} of the journey.",
                f"Indeed, {position} of this route, there is a portion where there are precisely {text_target} lanes.",
                f"As you travel this route, you will encounter instances with {text_target} lanes {position}.",
                f"There is a street {position} on this route with {text_target} lanes.",
                f"You will find {text_target} lane streets {position} on this route.",
                f"During the trip, you will come across a road with {text_target} lanes {position}.",
                f"Yes, you will encounter a {text_target} lane street {position} on this journey.",
                f"It is possible to find a section with {text_target} lanes {position} along this route.",
                f"The route features stretches with {text_target} lanes {position}.",
                f"Segments of the route have {text_target} lanes {position}.",
                f"You will encounter roads with {text_target} lanes {position} during your journey.",
                f"Expect to see {text_target} lane roads {position} along this route.",
                f"There is a part of the route {position} where the number of lanes equals {text_target}.",
                f"This route includes roads with {text_target} lanes {position}.",
                f"Streets with {text_target} lanes will be encountered {position} while traveling this route.",
                f"Pass sections with {text_target} lanes {position} on this route.",
                f"Traverse roads with {text_target} lanes {position} along the way.",
                f"On this particular route, encounter a road with {text_target} lanes {position}.",
                f"While traveling along this route, you will come across stretches of road that possess {text_target} lanes {position}.",
                f"In this particular route, you will find that all sections maintain a consistent number of lanes. As such, you can expect to traverse roads with exactly {text_target} lanes {position} of your journey, providing a uniform driving experience.",
                f"As you travel along this route, it's worth noting that there are no deviations in the number of lanes present. The entire journey is characterized by roads with exactly {text_target} lanes, ensuring a consistent and predictable driving environment {position}.",
                f"Throughout your entire journey on this route, you'll notice that there is a remarkable consistency in the infrastructure. Every segment and section features roads with exactly {text_target} lanes, maintaining a uniform experience for drivers {position} of the journey.",
                f"During your journey on this specific route, you will find that the number of lanes remains constant, providing a harmonious driving experience. The route is characterized by roads with precisely {text_target} lanes, which are designated paths for vehicles to travel in the same direction. Lanes play a crucial role in organizing traffic, promoting safety, and allowing for efficient movement of vehicles. In this particular instance, the consistent presence of {text_target} lanes {position} of the route ensures a uniform driving environment, aiding in navigation and traffic management. This consistency can be beneficial for drivers who appreciate predictability and well-structured roadways during their travels.",
            ]

        else:
            reply_list = [
                f"Yes, this route does have sections with the exact {text_target} lanes throughout the journey.",
                f"Indeed, this route features a portion where there are precisely {text_target} lanes.",
                f"As you travel this route, you will encounter instances with {text_target} lanes.",
                f"There is a street on this route with {text_target} lanes.",
                f"You will find {text_target} lane streets on this route.",
                f"During the trip, you will come across a road with {text_target} lanes.",
                f"Yes, you will encounter a {text_target} lane street on this journey.",
                f"It is possible to find a section with {text_target} lanes along this route.",
                f"The route features stretches with {text_target} lanes.",
                f"Segments of the route have {text_target} lanes.",
                f"You will encounter roads with {text_target} lanes during your journey.",
                f"Expect to see {text_target} lane roads along this route.",
                f"There is a part of the route where the number of lanes equals {text_target}.",
                f"This route includes roads with {text_target} lanes.",
                f"Streets with {text_target} lanes will be encountered while traveling this route.",
                f"Pass sections with {text_target} lanes on this route.",
                f"Traverse roads with {text_target} lanes along the way.",
                f"On this particular route, encounter a road with {text_target} lanes.",
                f"While traveling along this route, you will come across stretches of road that possess {text_target} lanes.",
                f"In this particular route, you will find that all sections maintain a consistent number of lanes. As such, you can expect to traverse roads with exactly {text_target} lanes throughout your journey, providing a uniform driving experience.",
                f"As you travel along this route, it's worth noting that there are no deviations in the number of lanes present. The entire journey is characterized by roads with exactly {text_target} lanes, ensuring a consistent and predictable driving environment.",
                f"Throughout your entire journey on this route, you'll notice that there is a remarkable consistency in the infrastructure. Every segment and section features roads with exactly {text_target} lanes, maintaining a uniform experience for drivers from start to finish.",
                f"During your journey on this specific route, you will find that the number of lanes remains constant, providing a harmonious driving experience. The route is characterized by roads with precisely {text_target} lanes, which are designated paths for vehicles to travel in the same direction. Lanes play a crucial role in organizing traffic, promoting safety, and allowing for efficient movement of vehicles. In this particular instance, the consistent presence of {text_target} lanes throughout the route ensures a uniform driving environment, aiding in navigation and traffic management. This consistency can be beneficial for drivers who appreciate predictability and well-structured roadways during their travels.",
            ]

        if use_cot:
            raise NotImplementedError("No template for chain-of-thought!")

        return random.choice(reply_list)

    def _get_specific_response_false(self, segments, target_value, use_cot):
        reply_list = [
            f"No, this route does not have any sections with {round(target_value)} lanes.",
            f"I'm sorry, but there are no portions with exactly {round(target_value)} lanes on this route.",
            f"Throughout this route, there are no instances where the number of lanes is {round(target_value)}.",
            f"There isn't a street with {round(target_value)} lanes on this route.",
            f"No {round(target_value)} lane streets can be found on this route.",
            f"At no point during the trip will we drive on a road with {round(target_value)} lanes.",
            f"There won't be any {round(target_value)} lane streets on this route.",
            f"Unfortunately, you won't find a section with {round(target_value)} lanes on this route.",
            f"The route does not feature any stretches with {round(target_value)} lanes.",
            f"No segments of the route have {round(target_value)} lanes.",
            f"We won't encounter any roads with {round(target_value)} lanes during our journey.",
            f"You should not expect to see any {round(target_value)} lane roads along this route.",
            f"There are no parts of the route where the number of lanes equals {round(target_value)}.",
            f"This route does not include any roads with {round(target_value)} lanes.",
            f"We won't come across any streets with {round(target_value)} lanes while traveling this route.",
            f"No sections with {round(target_value)} lanes will be passed on this route.",
            f"We're not going to traverse any roads with {round(target_value)} lanes along the way.",
            f"On this route, there is no specific segment or section featuring a {round(target_value)} lane road.",
            f"While traveling this route, we won't come across any stretches of road with {round(target_value)} lanes.",
        ]

        if use_cot:
            raise NotImplementedError("No template for chain-of-thought!")

        return random.choice(reply_list)

    def _categories_to_text(self, unique_categories, position=None):
        if not hasattr(self, "task_mode"):
            raise ValueError("Did not call get_task before!")

        if self.task_mode == "max":
            result_value = round(unique_categories.max())

            if position is None:
                reply_list = [
                    f"The maximum number of lanes is {result_value}.",
                    f"Given the route that was provided, the maximum number of lanes is {result_value}.",
                    f"On this journey you will experience a maximum of {result_value} lanes.",
                    f"The highest number of lanes you'll encounter on this route is {result_value}.",
                    f"At the widest point of your trip, there will be {result_value} lanes.",
                    f"During your travel, the maximum number of lanes you'll come across is {result_value}.",
                    f"Throughout your journey, the most lanes you'll find at any given point is {result_value}.",
                    f"The greatest number of lanes on this route is {result_value}.",
                    f"Your route has a maximum of {result_value} lanes at its widest point.",
                    f"At its peak, the route features {result_value} lanes.",
                    f"The route you're taking has a maximum of {result_value} lanes.",
                    f"During your travel along this route, you will encounter various road widths and lane configurations. The maximum number of lanes you'll find at any given point is {result_value}, which may reflect different driving experiences and traffic conditions depending on the specific location.",
                    f"On this journey, you'll come across a range of road conditions and lane counts. The highest number of lanes present on this route is {result_value}. This may result in diverse traffic situations, from quieter stretches to busier areas, depending on the context and location.",
                    f"As you make your way through this path, be prepared to experience different lane counts depending on the segment of the route. At its widest point, the route features {result_value} lanes, offering a varying driving experience and accommodating different levels of traffic.",
                    f"Throughout your entire trip, you might encounter various lane counts. However, the highest number of lanes you'll come across is {result_value}.",
                    f"Taking into consideration the entire journey, it's important to note that the maximum lane count at any specific point is {result_value}. Be prepared for this during your travel.",
                ]
            else:
                reply_list = [
                    f"The maximum number of lanes {position} the journey is {result_value}.",
                    f"Given the route that was provided, the maximum number of lanes {position} the route is {result_value}.",
                    f"On this journey, you will experience a maximum of {result_value} lanes {position}.",
                    f"The highest number of lanes you'll encounter {position} of this route is {result_value}.",
                    f"At the widest point of your trip {position}, there will be {result_value} lanes.",
                    f"During your travel, the maximum number of lanes you'll come across {position} is {result_value}.",
                    f"Throughout your journey, the most lanes you'll find {position} at any given point is {result_value}.",
                    f"The greatest number of lanes {position} this route is {result_value}.",
                    f"Your route has a maximum of {result_value} lanes at its widest point {position}.",
                    f"At its peak {position}, the route features {result_value} lanes.",
                    f"The route you're taking has a maximum of {result_value} lanes {position}.",
                    f"During your travel along this route, you will encounter various road widths and lane configurations. The maximum number of lanes you'll find {position} at any given point is {result_value}, which may reflect different driving experiences and traffic conditions depending on the specific location.",
                    f"On this journey, you'll come across a range of road conditions and lane counts {position}. The highest number of lanes present on this route is {result_value}. This may result in diverse traffic situations, from quieter stretches to busier areas, depending on the context and location.",
                    f"As you make your way through this path, be prepared to experience different lane counts depending on the segment of the route {position}. At its widest point, the route features {result_value} lanes, offering a varying driving experience and accommodating different levels of traffic.",
                    f"Throughout your entire trip, you might encounter various lane counts. However, the highest number of lanes you'll come across {position} is {result_value}.",
                    f"Taking into consideration the entire journey, it's important to note that the maximum lane count at any specific point {position} is {result_value}. Be prepared for this during your travel.",
                ]
        else:
            result_value = round(unique_categories.min())

            if position is None:
                reply_list = [
                    f"The minimum number of lanes is {result_value}.",
                    f"Given the route that was provided, the minimum number of lanes is {result_value}.",
                    f"On this journey, you will experience a minimum of {result_value} lanes.",
                    f"The lowest number of lanes you'll encounter on this route is {result_value}.",
                    f"At the narrowest point of your trip, there will be {result_value} lanes.",
                    f"During your travel, the minimum number of lanes you'll come across is {result_value}.",
                    f"Throughout your journey, the least lanes you'll find at any given point is {result_value}.",
                    f"The smallest number of lanes on this route is {result_value}.",
                    f"Your route has a minimum of {result_value} lanes at its narrowest point.",
                    f"At its least, the route features {result_value} lanes.",
                    f"The route you're taking has a minimum of {result_value} lanes.",
                    f"During your travel along this route, you will encounter various road widths and lane configurations. The minimum number of lanes you'll find at any given point is {result_value}, which may reflect different driving experiences and traffic conditions depending on the specific location.",
                    f"On this journey, you'll come across a range of road conditions and lane counts. The lowest number of lanes present on this route is {result_value}. This may result in diverse traffic situations, from busier stretches to quieter areas, depending on the context and location.",
                    f"As you make your way through this path, be prepared to experience different lane counts depending on the segment of the route. At its narrowest point, the route features {result_value} lanes, offering a varying driving experience and accommodating different levels of traffic.",
                    f"Throughout your entire trip, you might encounter various lane counts. However, the lowest number of lanes you'll come across is {result_value}.",
                    f"Taking into consideration the entire journey, it's important to note that the minimum lane count at any specific point is {result_value}. Be prepared for this during your travel.",
                ]
            else:
                reply_list = [
                    f"The minimum number of lanes {position} the journey is {result_value}.",
                    f"Given the route that was provided, the minimum number of lanes {position} the route is {result_value}.",
                    f"On this journey, you will experience a minimum of {result_value} lanes {position}.",
                    f"The lowest number of lanes you'll encounter {position} of this route is {result_value}.",
                    f"At the narrowest point of your trip {position}, there will be {result_value} lanes.",
                    f"During your travel, the minimum number of lanes you'll come across {position} is {result_value}.",
                    f"Throughout your journey, the least lanes you'll find {position} at any given point is {result_value}.",
                    f"The smallest number of lanes {position} this route is {result_value}.",
                    f"Your route has a minimum of {result_value} lanes at its narrowest point {position}.",
                    f"At its least {position}, the route features {result_value} lanes.",
                    f"The route you're taking has a minimum of {result_value} lanes {position}.",
                    f"During your travel along this route, you will encounter various road widths and lane configurations. The minimum number of lanes you'll find {position} at any given point is {result_value}, which may reflect different driving experiences and traffic conditions depending on the specific location.",
                    f"On this journey, you'll come across a range of road conditions and lane counts {position}. The lowest number of lanes present on this route is {result_value}. This may result in diverse traffic situations, from busier stretches to quieter areas, depending on the context and location.",
                    f"As you make your way through this path, be prepared to experience different lane counts depending on the segment of the route {position}. At its narrowest point, the route features {result_value} lanes, offering a varying driving experience and accommodating different levels of traffic.",
                    f"Throughout your entire trip, you might encounter various lane counts. However, the lowest number of lanes you'll come across {position} is {result_value}.",
                    f"Taking into consideration the entire journey, it's important to note that the minimum lane count at any specific point {position} is {result_value}. Be prepared for this during your travel.",
                ]

        return random.choice(reply_list)
