from routellm.dataset.verbalization.tasks.cot_base_task import COTBaseTask


class WeatherTask(COTBaseTask):
    NAME = "llm-tasks.weather-task"
    VERSION = 0
    TASK_LIST = [
        "Can you please create a short weather report of this route",
        "Please give me a one sentence weather report of the following journey",
        "Can you please go into how the weather will be on this route",
        "Given the following route, how is the weather",
        "Is the weather good on this route and why",
        "What are the indications for good weather in this route",
        "Does the weather present in this route make driving dangerous",
        "Does the weather in this route make driving difficult and why",
        "Is this dangerous weather to be driving in? Why",
        "How do the current weather conditions impact visibility on this route",
        "What types of weather-related obstacles can I expect along this journey",
        "What types of weather-related obstacles can I expect along this journey. I am taking my car",
        "What types of weather-related obstacles can I expect along this journey. I am walking",
        "What types of weather-related obstacles can I expect along this journey. I am taking public transportation",
        "Are there any areas prone to severe weather conditions on this route",
        "Can you provide a weather forecast for my upcoming trip",
        "What's the likelihood of encountering heavy rain or snow during my journey",
        "Are strong winds expected on any parts of this route, and how may they affect driving",
        "Does the weather forecast suggest potential road closures along my travel path",
        "Considering the current climate, what should I pack in case of a weather emergency while driving through this route",
        "Which sections of the route might require extra caution due to poor weather conditions such as flooding or icy roads",
        "If I were to personify the weather along this journey, what kind of mood would it be in and why",
        "Imagine you're a local weather reporter; how would you poetically describe the current climate on this route",
        "If my trip's weather forecast was a movie genre, which one would it be and how will that affect my driving experience",
        "Would people like the weather on this route",
        "Is the weather at the start suitable for any activities outside and why",
        "I am from a really warm region. Is there anything I should prepare when going on this trip",
        "Weatherwise, are there any reasons why I should not drive this route",
        "Is the weather on this route dangerous",
        "Are there any weather-related dangers on this path",
        "I really hate rain. Would I like this route",
        "Would people generally like the weather on this portion of the route",
        "What would people describe the weather on this route as",
    ]

    NON_TEXT_COLUMNS = [
        "oneway",
        "ref",
        "reversed",
        "bridge",
        "tunnel",
        "free_flow_speed",
        # "bearing",
        # "length",
        "travel_time",
        "has_junction",
        "name",
        "lanes",
        "incident_certainty",
        "highway",
        "landuse",
        "speed_kph",
    ]

    def get_system_message(self, style_info=None):
        text, style = super().get_system_message(style_info=style_info)

        return f"{text} Always answer in full sentences.", style
