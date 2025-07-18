from routellm.dataset.verbalization.tasks.cot_base_task import COTBaseTask


class BeautyTask(COTBaseTask):
    NAME = "llm-tasks.beauty-task"
    VERSION = 0
    TASK_LIST = [
        "Would drivers generally perceive this route as beautiful and why",
        "Please share your thoughts: would this trip generally be perceived as beautiful and appealing to drivers and their passengers? Please explain why",
        "Given the following route, would the general public perceive it as a scenic driving route. Can you also explain why",
        "Do you think this route would be considered visually appealing while driving to most people? Please provide reasons",
        "How likely is it that this route would be regarded as picturesque by the majority of drivers? Please elaborate in full sentences",
        "In your opinion, would this journey be seen as aesthetically pleasing by most individuals? Kindly explain your reasoning, keeping in mind that the means of transportation is a car",
        "Considering driving on his route, how appealing do you think it would be to the average person in terms of natural beauty? Please justify your answer",
        "To what extent would traveling by car on this path be considered scenic or charming by the general population? Please clarify your thoughts",
        "How would you rate the overall attractiveness of this route for cars in the eyes of most people? Please provide an explanation",
        "Is it likely that people would find this route to be visually captivating while driving on it? Please offer some reasons for your opinion",
        "In terms of beauty, how well do you think this route would be received by the majority of drivers? Kindly expound on your thoughts",
        "Would this route be considered alluring or striking to most drivers or their passengers? Please support your answer with reasons in full sentences",
        "Do you believe this route would be deemed enchanting or picturesque by the general public? The mode of transportation is a car. Please elucidate your perspective in full sentences",
        "Considering the surrounding landscape and environment, how would you rate the overall beauty of this route for drivers? Please explain your evaluation",
        "Taking into account the type of area around the street, do you think this route would be deemed visually appealing by most traffic participants? Kindly provide reasons for your opinion",
        "Given the nature of the nearby surroundings, how would you assess this route's potential to be seen as scenic or picturesque? The user in mind is driving in a car. Please elaborate on your thoughts in full sentences",
        "In light of the surrounding area and its features, how likely is it that this route would be considered charming or attractive to the majority of drivers? Please justify your answer in full sentences",
        "With respect to the environment and landscape around the route, how well do you think it would be received in terms of aesthetic appeal by most drivers and their passengers? Kindly expound on your perspective",
        "Considering the potential traffic incidents along this route, how would you evaluate its overall beauty or appeal to most drivers? Please provide your reasoning",
        "Taking into account any traffic incidents that might occur on the route, do you think its attractiveness towards motorists would be affected? Kindly explain your thoughts",
        "Given the possibility of traffic issues along this path, how do you think the overall scenic quality of the route would be perceived by most traffic participants, especially drivers? Please elaborate",
        "A user travels on this path in a car. In light of potential traffic disturbances on this route, how likely is it that the route's charm or visual appeal would be impacted? Please justify your answer in full sentences",
        "With respect to any traffic incidents that may arise along the route, how well do you think its aesthetic allure would be maintained in the eyes of the drivers? Kindly expound on your perspective",
    ]

    NON_TEXT_COLUMNS = [
        "oneway",
        "ref",
        "reversed",
        "bridge",
        "tunnel",
        "free_flow_speed",
        "bearing",
        # "length",
        "travel_time",
        "has_junction",
        "wind_speed",
        "name",
        "lanes",
    ]
