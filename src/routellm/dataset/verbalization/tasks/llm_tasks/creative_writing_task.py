import random

from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks.cot_base_task import COTBaseTask


def format_num(number):
    num_map = {
        1: ["a", "one", "1"],
        2: ["two", "2"],
        3: ["three", "3"],
        4: ["four", "4"],
        5: ["five", "5"],
    }

    return random.choice(num_map[number])


class CreativeWritingTask(COTBaseTask):
    # Inspired by LLaVA
    NAME = "llm-tasks.creative-writing-task"
    VERSION = 0
    MAX_ROUTE_TOKENS = 200
    MIN_SEQUENCE_LENGTH = 50

    MIN_NUM_SPEAKER = 2
    MAX_NUM_SPEAKER = 4
    TARGET_AUDIENCES = [
        "Setup: Two neighbors, a young professional and a retired teacher, are sitting inside a car, discussing a part of the route they will be driving, as they prepare for a shared journey to attend a community event.",
        "Setup: A group of friends with diverse backgrounds - a graphic designer, a nurse, and a software engineer - is waiting in a parking lot, planning their next adventure in their off-road vehicle.",
        "Setup: A couple, one being a travel blogger and the other a photographer, are sitting inside their camper van, discussing the ideal route to explore and document their upcoming road trip.",
        "Setup: A father, who is a mechanic, and his teenage daughter, who has recently learned to drive, are sitting in their family car in the driveway, discussing whether this is the safest and most efficient route for her daily school commute.",
        "Setup: Three colleagues from different departments - marketing, finance, and IT - are carpooling and waiting at their office's parking lot. They're discussing the quickest route to reach a team-building event organized by their company.",
        "Setup: A university student who just got their driver's license is sitting in their compact car with a fellow student, discussing this part of the route that they'll use to reach their off-campus study group meeting.",
        "Setup: A proud sports car owner and their excited friend are sitting inside the vehicle, planning a thrilling drive to showcase the car's capabilities and performance.",
        "Setup: A family with a comfortable family car is preparing for an outing, with the parents discussing whether this is the most kid-friendly route to reach their destination while their children wait in the backseat.",
        "Setup: A driving instructor and their nervous student, who does not know how to drive yet, are sitting in a dual-control car, discussing this route to practice driving in different weather conditions.",
        "Setup: A group of friends with a mix of experienced and newly licensed drivers is gathered around a convertible, planning a road trip and discussing what makes this the most scenic route to get there.",
        "Setup: A couple with a dislike for bad weather is sitting in their all-wheel-drive SUV, discussing a part of the journey their drive to visit family.",
        "Setup: A family with adult children is sitting in their minivan, preparing for a long drive to attend a family reunion, and discussing everyone's preferences and needs regarding to the journey.",
        "Setup: A group of high school friends who haven't seen each other in years is meeting up in a parking lot, getting ready to embark on a nostalgic road trip while discussing this route and their favorite hangout spots from the past.",
        "Setup: Two coworkers, one experienced driver and another who recently obtained their license, are sitting in a company car, driving to a client meeting while considering traffic patterns and road conditions.",
        "Setup: A parent who is teaching their teenager, who just got their driver's license, about safe driving habits is sitting in the passenger seat while discussing if this is the most appropriate route for practicing different driving skills.",
        "Setup: A group of photography enthusiasts is sitting in a nice new sports car, discussing the most picturesque route for capturing the beauty of the landscape during their upcoming photography expedition.",
        "Setup: A carpool group with members who prioritize safety is waiting at their designated pickup location, discussing the safety of this route while considering factors such as road conditions and accident-prone areas.",
        "Setup: A couple planning a romantic weekend getaway is sitting in their car, discussing the most scenic route to reach their destination, with a focus on beautiful landscapes.",
        "Setup: A family with young children is sitting in their car, preparing for a long drive to a vacation destination, and discussing the best route to avoid potential traffic congestion and minimize travel time.",
        "Setup: Two friends with a shared interest in meteorology are sitting in a weather-resistant vehicle, discussing the ideal route for chasing storms and observing weather phenomena during their upcoming adventure.",
    ]
    TASK_LIST = [
        *[
            f"{setup}\nWrite the dialog regarding this part of the route"
            for setup in TARGET_AUDIENCES
        ],
        *[
            f"{setup}\nThis is a part of the journey. Write their conversation"
            for setup in TARGET_AUDIENCES
        ],
        *[
            f"This is a setup for a conversation about the following route. Please write it: {setup}"
            for setup in TARGET_AUDIENCES
        ],
        *[
            f"Consider this description: {setup}\nWrite a matching dialogue considering this part of the route"
            for setup in TARGET_AUDIENCES
        ],
        *[
            f"{setup}\nHow could a potential conversation go, if they were talking about this part of the trip"
            for setup in TARGET_AUDIENCES
        ],
    ]

    NON_TEXT_COLUMNS = []

    def get_task(self):
        punctuation_characters = [
            "?",
            ".",
            ":",
            "",
            " ",
            "\n",
            "\n\n",
        ]

        return random.choice(self.TASK_LIST) + random.choice(punctuation_characters)

    def get_system_message(self, style_info=None):
        system_message = "You are a talented dialogue writer, adept at crafting realistic and engaging conversations based on short descriptions. They utilize a diverse vocabulary, ensuring each character sounds authentic and natural. Praised for their precision and relevance, their dialogues maintain the perfect length and intrigue, skillfully omitting unnecessary details from the description. Make it sound like a real conversation. Most people would not talk about exact lengths or grades and so on. Do not mention any values from the route that are not necessary for the conversation. Usually streets are flat, steep, very steep, hilly and so on before mentioning the exact grade. When talking about the current speed, people also tend to be vague. Also when talking about the wind speed and the speed of wind gust, the delay through an incident and the travel time people tend to be vague. Usually the travel time is rounded to minutes or even hours if it is a longer trip. Winds are usually only categorized in how strong they are, i.e. a breeze, strong winds, . The delay caused by a traffic incident is usually also estimated in minutes."  # n AI assistant that accelerates at writing realistic, capturing and interesting dialogs.

        if style_info is None:
            style_text, style = self.get_random_style()
        else:
            style = style_info
            style_text = get_style(**style_info)

        return (
            f"{system_message} Follow these phrasing instructions: {style_text}",
            style,
        )  # f"{system_message} Follow these style instructions: {style_text}", style
