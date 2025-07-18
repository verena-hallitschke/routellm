import random

from routellm.dataset.verbalization.tasks.boolean_base_task import BooleanBaseTask


class IncidentReportedTask(BooleanBaseTask):
    NAME = "feature-tasks.incident-reported-task"
    VERSION = 0
    TASK_LIST = [
        "Hey, I am driving on this route. Have there been any reports of traffic incidents",
        "I am currently traveling on this path. Are there any known traffic incidents",
        "I'm driving along this route. Have any traffic-related events been reported",
        "Greetings, I am on this course right now. Have there been instances of traffic issues",
        "Hi, I'm taking this route at the moment. Are there reports of any accidents or delays",
        "Hey there, I'm following this road right now. Has anyone reported any traffic problems",
        "Hello, as I drive on this particular route, have there been reports of any traffic disturbances",
        "Hi there, I'm navigating this way presently; have any incidents affecting traffic been mentioned",
        "Greetings! While driving on this route, are you aware of any reported occurrences of traffic disruptions",
        "On my current journey using this path, has anyone shared information about possible traffic complications",
        "Hello! As I progress along this route, are you informed about any existing reports regarding traffic situations",
        "Are there any known traffic-related events reported",
        "Have any instances of traffic issues been mentioned",
        "Are there reports of any accidents or delays in traffic",
        "Has anyone reported any problems with the traffic flow",
        "Have there been reports about disturbances affecting traffic",
        "Have incidents impacting traffic been communicated recently",
        "Are you aware of any reported occurrences of traffic disruptions",
        "Has information about possible traffic complications been shared lately",
        "Are there existing reports regarding troubling traffic situations",
        "Have updates about incidents concerning traffic conditions been provided",
        "Has anyone mentioned any traffic-related issues recently",
        "Were there notifications of any incidents involving traffic",
        "Have you come across updates on traffic accidents or delays",
        "Are there any reported cases of problems affecting the roads",
        "Did anyone report mishaps or disruptions in traffic flow lately",
        "Has information surfaced about events causing traffic congestion",
        "Were there any recent accounts of incidents impacting transportation",
        "Have you heard about reported happenings that could hinder the traffic",
        "Is there any news concerning situations that might affect road conditions",
        "Have people been talking about occurrences related to disrupted travel",
        "Are there any signs of reported traffic issues along this path",
        "Have you noticed any evidence of traffic incidents on this route",
        "Is there any indication that traffic problems have been reported for this journey",
        "Are we aware of any clues pointing towards reported accidents on this road",
        "Have there been hints suggesting the presence of traffic events on this course",
        "Can you spot any signals about potential traffic disturbances while traveling this way",
        "Do we have any information hinting at possible reported incidents along this drive",
        "Has there been mention of markers indicating traffic complications on this route",
        "Are there signs available about known issues affecting the conditions of this road",
        "Is there any data pointing towards reports of setbacks during transit on this path",
        "Any traffic issue indicators",
        "Noticed incident signs on route",
        "Evidence of reported problems",
        "Reports of accidents on this road",
        "Signs of known issues on road",
        "Any traffic incident reports",
        "Reports of traffic issues",
        "Traffic problem updates",
        "Known incidents reported",
        "Accident alerts received",
        "Trouble on roads reported",
        "Congestion reports shared",
        "Have any noteworthy reports surfaced regarding accidents, delays, or disruptions in the traffic flow",
        "Are we aware of any comprehensive information detailing occurrences of issues related to road conditions or traffic congestion",
        "Has anyone compiled or shared updates concerning instances of vehicular mishaps, collisions, or other traffic-related challenges",
        "Can we find evidence of elaborate statements on incidents affecting the smooth flow and operation of transportation networks recently",
        "Have authorities or individuals communicated any significant revelations about complex situations stemming from problematic traffic events",
    ]

    ONE_SEGMENT_TRUE_TEMPLATE = [
        *[
            f"Yes, there is an incident that was reported along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are reported incidents.",
        *[
            f"Yes, you might encounter incidents, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are incidents that were reported on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
    ]
    FALSE_TEMPLATE = [
        *[
            f"No, there is no reported incidents along the {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        "There are no documented traffic incidents.",
        *[
            f"No, you will encounter no reported incidents, when traveling along the given {synonym}."
            for synonym in ["route", "road", "path"]
        ],
        *[
            f"There are no reported incidents on this {synonym}."
            for synonym in ["route", "road", "path", "journey"]
        ],
        "No, no road incidents are encountered.",
    ]
    FEATURE_NAME = "incident_reported"

    def _get_multi_segment_response(
        self, plural: bool, return_position: bool, position: str
    ):
        template = [
            *[
                f"Yes, there {'is an incident that was reported' if not plural else 'are multiple reported incidents'} along the {synonym}.{f' They are located {position}.' if return_position and plural else f' It is located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            f"There {'are reported traffic incidents' if plural else 'is a reported incident'}{'' if not return_position else f' {position}'}.",
            *[
                f"Yes, since there have been reports, you might encounter {'traffic incidents' if plural else 'a traffic incident'}, when traveling along the given {synonym}.{f' They are located {position}.' if return_position else ''}"
                for synonym in ["route", "road", "path"]
            ],
            *[
                f"There {'are traffic traffic incidents that have been reported' if plural else 'is a report of a traffic incident'} on this {synonym}{f', which are located {position}' if return_position and plural else f', which is located {position}' if return_position else ''}."
                for synonym in ["route", "road", "path", "journey"]
            ],
        ]

        return random.choice(template)
