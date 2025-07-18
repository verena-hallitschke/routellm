from routellm.dataset.verbalization.tasks.cot_base_task import COTBaseTask


class TrafficTask(COTBaseTask):
    NAME = "llm-tasks.traffic-task"
    VERSION = 0
    TASK_LIST = [
        "Can you please create a short traffic report of this route",
        "Would people like the traffic on this route",
        "Traffic-wise, are there any reasons why I should not drive this route",
        "Is the traffic on this route dangerous",
        "Are there any traffic-related dangers on this path",
        "I really hate traffic jams. Would I like this route",
        "Would people generally like the traffic on this portion of the route",
        "What would people describe the traffic on this route as",
        "What are the common causes of traffic congestion on this route",
        "How does the time of day affect the traffic flow on this route",
        "Can you provide me with real-time updates on the current traffic situation for this journey",
        "Which segments of this route experience the heaviest traffic, and how can they be avoided",
        "How does the overall travel time change due to average peak-hour congestion levels along this specific route",
        "If this route was a blockbuster movie, what would its traffic-related title be and why",
        "Imagine that the traffic on this route had a personality: how would you describe it, and what kind of conversation could I expect to have with it",
        "If famous artists were to depict the traffic on this route in their signature styles, whose artwork would best represent the journey's experience",
        "How would you rate your overall satisfaction with the traffic conditions on this route, and why do you feel that way",
        "What are the most common sentiments or emotions people express when discussing the traffic on this journey",
        "If given a choice, what changes would people make to improve their experience with the traffic along this route",
        "How does navigating through the traffic on this route impact people's stress levels or mood throughout their day",
        "Can you share any anecdotes or stories from individuals who have experienced notable incidents while dealing with the traffic on this particular route",
        "Hey, what would folks say about the traffic on this road? Is it a smooth ride or a total headache",
        "Do people feel chill when driving this route, or does it make them stressed out and grumpy",
        "What precautions should I take before driving this route to avoid any potential traffic-related hazards",
        "Are there any specific areas along this journey that are have accidents or other dangers that I should be aware of",
        "How can I best prepare myself for unexpected traffic situations or challenges that may arise while navigating this route",
        "Regarding traffic, will this be a smooth ride",
        "Talking traffic, will this route be enjoyable",
    ]

    NON_TEXT_COLUMNS = [
        "oneway",
        "ref",
        "reversed",
        "bridge",
        "tunnel",
        "bearing",
        # "length",
        "has_junction",
        "name",
        "landuse",
        "precipitation",
        "wind_speed",
        "lightning",
        "grade",
        "curvature",
    ]

    def get_system_message(self, style_info=None):
        text, style = super().get_system_message(style_info=style_info)

        return f"{text} Always answer in full sentences.", style
