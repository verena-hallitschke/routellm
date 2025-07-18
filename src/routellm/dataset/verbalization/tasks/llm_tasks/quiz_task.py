import random

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


class QuizTask(COTBaseTask):
    NAME = "llm-tasks.quiz-task"
    VERSION = 0
    MAX_ROUTE_TOKENS = 100

    MIN_NUM_ANSWERS = 1
    MAX_NUM_ANSWERS = 3
    TASK_LIST = [
        *[
            f"Can you please ask {format_num(num)} question{'' if num == 1 else 's'} about this route and also provide {'an ' if num == 1 else ''}answer{'' if num == 1 else 's'}"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"I want to design an exam where students have to answer question{'' if num == 1 else 's'} about this route. Can you help me?\nPlease give me {format_num(num)} tasks and their answer(s)"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        "Can you give me one task and the answer to it",
        "Please give me one possible question regarding the traffic of the route and the matching answer",
        "Can you give me one task and the answer to it regarding the weather on this route",
        "Can you give me some questions and answers regarding the bridges on this route",
        "I'd like a couple of questions about this route. Also add the correct solution",
        "Create a trivia question with solution based on the following route description",
        "Using the route description provided, formulate a trivia quiz with multiple choice. Add the correct answers at the end",
        "Given this route description, develop a themed multiple-choice quiz with correct answers related to it",
        "Based on the route information below, please devise an inspired question with answer",
        "Considering the route description, please come up with a question and reply suitable for a quiz",
        "Utilizing the route details provided, generate a trivia-style question with reply",
        "Incorporate the route description into a quiz-like question and response",
        "Employ the route information given to construct a trivia-based question",
        "With the route description in mind, please formulate a question and a solution fitting for a quiz",
        "Taking the route description into account, create a question reminiscent of a trivia game. Also provide the correct answer",
        *[
            f"Create {format_num(num)} question{'' if num == 1 else 's'} and answer{'' if num == 1 else 's'} focusing on the weather conditions along this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Formulate {format_num(num)} trivia question{'' if num == 1 else 's'} with correct replies related to the traffic on this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Devise {format_num(num)} query(ies) based on the curvature of the road in this route description. Also provide answer(s)"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Please come up with {format_num(num)} question{'' if num == 1 else 's'} (and the answers) about the length of this particular route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate {format_num(num)} question{'' if num == 1 else 's'} and answer{'' if num == 1 else 's'} regarding the junctions found along this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Ask {format_num(num)} question{'' if num == 1 else 's'} about traffic incidents that may occur on this route. Add the response/responses at the end"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Inquire about the time of day and its effects on this route with {format_num(num)} question{'' if num == 1 else 's'} and also provide replies"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Create {format_num(num)} question-reply pair{'' if num == 1 else 's'} related to the current traffic speed on this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Formulate {format_num(num)} query(ies) about the free flow speed of this route and also add the correct replies"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Ask {format_num(num)} question{'' if num == 1 else 's'} and the generate the correct solutions focusing on the usage of areas surrounding this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Devise {format_num(num)} question{'' if num == 1 else 's'} about the speed limit enforced on this route. Also provide the correct solutions"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Please come up with {format_num(num)} question-solution-pair{'' if num == 1 else 's'} regarding the bridges along this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate {format_num(num)} question{'' if num == 1 else 's'} with answers about tunnels found on this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Inquire with {format_num(num)} question{'' if num == 1 else 's'} about the type of road that makes up this route. Answer them as well"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Create a quiz with {format_num(num)} question{'' if num == 1 else 's'} and matching answer(s) concerning the lightning conditions along this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Formulate {format_num(num)} question{'' if num == 1 else 's'} about the cloud coverage on this route and answer"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate and answer {format_num(num)} query(ies) related to precipitation along this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Please come up with {format_num(num)} question{'' if num == 1 else 's'} and {'reply' if num == 1 else 'replies'} regarding the temperature on this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate {format_num(num)} question{'' if num == 1 else 's'} and solution{'' if num == 1 else 's'} about traffic incidents on this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Inquire about the certainty of these incidents with {format_num(num)} question{'' if num == 1 else 's'} and response{'' if num == 1 else 's'}"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Create and solve {format_num(num)} question{'' if num == 1 else 's'} about the magnitude of traffic incidents on this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Formulate and answer {format_num(num)} {'query' if num == 1 else 'queries'} regarding the delay caused by incidents on this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Come up with {format_num(num)} question{'' if num == 1 else 's'} about whether the street is a one-way street and provide the correct answer"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Design an exam with {format_num(num)} task(s) related to this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Create {format_num(num)} task{'' if num == 1 else 's'} focusing on the weather conditions along this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Devise {format_num(num)} task{'' if num == 1 else 's'} based on the curvature of the road in this route description, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Please come up with {format_num(num)} task{'' if num == 1 else 's'} about the length of this particular route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate {format_num(num)} task{'' if num == 1 else 's'} regarding the junctions found along this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Ask {format_num(num)} task{'' if num == 1 else 's'} about traffic incidents that may occur on this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Inquire about the time of day and its effects on this route with {format_num(num)} task{'' if num == 1 else 's'}, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Create a sheet with {format_num(num)} question{'' if num == 1 else 's'} related to various aspects of this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Formulate a quiz with {format_num(num)} trivia question{'' if num == 1 else 's'} that cover multiple features of this route. Add the quiz solution at the bottom"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Please come up with {format_num(num)} question{'' if num == 1 else 's'} that address the overall experience of traveling on this route. Answer"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate and solve {format_num(num)} question{'' if num == 1 else 's'} that encompass a broad range of topics about this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Ask {format_num(num)} question{'' if num == 1 else 's'} with answers that require a comprehensive understanding of this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Design an exam with {format_num(num)} task{'' if num == 1 else 's'} that cover a variety of topics about this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Create {format_num(num)} task{'' if num == 1 else 's'} that address multiple aspects of this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Please come up with {format_num(num)} task{'' if num == 1 else 's'} that explore the general characteristics of this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate {format_num(num)} task{'' if num == 1 else 's'} that require a holistic understanding of this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Ask {format_num(num)} task{'' if num == 1 else 's'} that challenge the knowledge of various features of this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Create a test with {format_num(num)} question{'' if num == 1 else 's'} that combine aspects of weather conditions and traffic incidents along this route. Also provide a solution sheet"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Formulate {format_num(num)} trivia question{'' if num == 1 else 's'} that cover both the junctions and the type of road in this route. Please answer as well"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Please come up with {format_num(num)} question{'' if num == 1 else 's'} and {'an ' if num == 1 else ''}answer{'' if num == 1 else 's'} addressing the speed limit and the current traffic speed on this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate and reply to {format_num(num)} question{'' if num == 1 else 's'} that focus on the curvature of the road and the presence of bridges or tunnels along this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Ask and answer {format_num(num)} question{'' if num == 1 else 's'} that require knowledge of the route's length, traffic incidents, and the associated delays"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Create {format_num(num)} question{'' if num == 1 else 's'} {'a ' if num == 1 else ''}response{'' if num == 1 else 's'} that combine the aspects of cloud coverage and temperature along this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Formulate and solve {format_num(num)} trivia question{'' if num == 1 else 's'} that cover both the free flow speed and the usage of areas surrounding this route"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Please come up with {format_num(num)} question{'' if num == 1 else 's'} addressing the precipitation and time of day on this route. Can you also give the solution"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate {format_num(num)} question{'' if num == 1 else 's'} that focus on the traffic incidents and the certainty on this route and add the correct results"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Ask {format_num(num)} question{'' if num == 1 else 's'} that require knowledge of the route's traffic speed, lighting conditions, and speed limits. Solve them"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Design {format_num(num)} task{'' if num == 1 else 's'} that explore the relationship between traffic incidents and junctions on this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Create {format_num(num)} task{'' if num == 1 else 's'} that address the effects of temperature and weather conditions on traffic flow, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Please come up with {format_num(num)} task{'' if num == 1 else 's'} that examine the connection between curvature, type of road, and traffic speed on this route, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Generate {format_num(num)} task{'' if num == 1 else 's'} that require understanding of the route's bridges, tunnels, and usage of surrounding areas, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Ask {format_num(num)} task{'' if num == 1 else 's'} that investigate the impact of weather conditions and time of day on traffic incidents and delays, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
        ],
        *[
            f"Ask {format_num(num)} riddle{'' if num == 1 else 's'} that regarding the attached journey, and provide the answers"
            for num in range(MIN_NUM_ANSWERS, MAX_NUM_ANSWERS)
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

        cot_list = [
            "Answer the question step by step to make it understandable how you came to your conclusion. ",
            "Give a step by step rundown. ",
            "Please answer step by step. ",
            "Please explain how you reached the conclusion step by step. ",
        ]

        prepend_list = [
            "Give an answer in every-day language. ",
            "Keeping in mind that this person is also driving a sports car. ",
            "Make sure to use a beautiful language. ",
            "Keep in mind that we are driving. ",
            "Use the same language a teacher would use. ",
            "Use the language a professor would use. ",
        ]
        cot_string = random.choice(
            cot_list
            if random.random() < 0.3
            else prepend_list
            if random.random() < 0.2
            else [""]
        )

        return (
            cot_string
            + random.choice(self.TASK_LIST)
            + random.choice(punctuation_characters)
        )
