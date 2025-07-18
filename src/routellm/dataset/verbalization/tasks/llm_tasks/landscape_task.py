# Questions regarding grade and curvature

from routellm.dataset.verbalization.tasks.cot_base_task import COTBaseTask


class LandscapeTask(COTBaseTask):
    NAME = "llm-tasks.landscape-task"
    VERSION = 0
    TASK_LIST = [
        "Would you describe this area as mountainous",
        "Are there any hills",
        "What is the landscape like",
        "Does this trip have a high curvature",
        "What are the most prominent features of the landscape on this ride",
        "Are there significant changes in elevation throughout the trip, like steep inclines or descents",
        "Are there any notable bodies of water along the route, such as lakes or rivers",
        "Do we pass through any forests or wooded areas during this car ride",
        "Considering the curvature and elevation, how challenging might this drive be for inexperienced drivers",
        "Based on the road type, would you recommend a specific type of vehicle for this journey",
        "How does the road's curvature affect the overall travel time or distance covered during this ride",
        "Do changes in elevation along the route require any special driving techniques or precautions",
        "Are there any known hairpin turns or switchbacks that drivers should be aware of on this route",
        "Is there a significant difference in road conditions between flat and elevated parts of the drive",
        "For those who may be prone to claustrophobia or fear of heights, is this a suitable journey",
    ]

    NON_TEXT_COLUMNS = [
        "oneway",
        "ref",
        "reversed",
        "bearing",
        # "length",
        "has_junction",
        "name",
        "precipitation",
        "wind_speed",
        "lightning",
        "incident_certainty",
    ]

    def get_system_message(self, style_info=None):
        text, style = super().get_system_message(style_info=style_info)

        return f"{text} Always answer in full sentences.", style
