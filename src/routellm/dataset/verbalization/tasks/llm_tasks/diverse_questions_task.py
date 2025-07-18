from routellm.dataset.verbalization.tasks.cot_base_task import COTBaseTask


class DiverseQuestionsTask(COTBaseTask):
    NAME = "llm-tasks.diverse-questions-task"
    VERSION = 0
    TASK_LIST = [
        "Considering the proximity to various area types (commercial, industrial, recreational, etc.) mentioned in each step, infer potential traffic patterns such as peak times or likely congestion points",
        "Analyze the data for lane changes and infer how these fluctuations might contribute to traffic flow and congestion at each step",
        "Calculate the total time lost due to traffic by comparing the segment free flow speed time and the current average traffic speed time for each step",
        "Summarize the areas around the street during the journey and what they are used for",
        "Explore the relationship between traffic speeds and the type of area close to the street",
        "Discuss how the presence of bridges and tunnels might complicate traffic flow and control",
        "Formulate a traffic advisory for commuters based on the current traffic and weather conditions",
        "Identify potential bottlenecks on the route and suggest interventions",
        "Analyze the effect of proximity to commercial areas on traffic patterns",
        "Discuss the potential impact of road curvature on traffic congestion",
        "Evaluate the adequacy of the number of lanes throughout the journey for smooth traffic",
        "Assess the impact of light conditions on visibility and traffic safety for the given journey",
        "Assess the overall traffic efficiency of the route given the various conditions",
        "Explain the role of variable speed limits in managing traffic under changing weather conditions",
        "How does the speed limit on this route effect the safety of the travelers",
        "How do sections without a speed limit impact the traffic in the route at hand",
        "At what time will we arrive at our destination",
        "Could you tell me, at what time we arrive at our destination (with and without traffic)",
        "During a this trip, can we identify any patterns in traffic density or speed fluctuations based on the weather conditions encountered along the route",
        "Are there time-dependent patterns, such as rush hour congestion, that occur during the trip",
        "Based on the observed patterns, can we estimate the most likely travel time for the entire route or for segments of it",
        "How do external factors such as time of day, day of the week, or seasonal changes affect the traffic pattern during the trip",
        "This route was recorded in fall. Can you tell me why",
        "Imagine you are tasked with designing an innovative app feature that uses our route measurement data to enhance the driving experience during a single trip. How would you utilize traffic patterns, weather conditions, and street types to create this feature, and what unique problem would it solve for the driver",
        "If you were to narrate a story about a day in the life of a commuter using the traffic and weather data in our route, what kind of character would you create, and what challenges would they face on their journey",
        "How could you turn our route measurement data into an engaging game or challenge for drivers? What would be the rules, objectives, and rewards",
        "Develop a unique marketing campaign that uses our traffic and weather data insights to promote carpooling during peak hours. What would be your campaign's tagline and key visual elements",
        "Design a personalized driving assistant that uses traffic and weather data to offer customized advice and options for each segment of the driver's route. What features would make this assistant truly helpful and unique? Create an example based on this route",
        "If you had the power to use our route data to make a positive social impact, what initiative would you launch, and how would it help the community",
        "Develop a strategy for a ride-sharing service to dynamically adjust its pricing and routing algorithms in response to sudden changes in weather conditions to maintain efficiency and safety. Use this as an example route",
        "Using the patterns identified in traffic flow and stop-and-go conditions, create a plan to help drivers improve their fuel efficiency. What driving behaviors or route adjustments would you recommend",
        "Analyze traffic volume and patterns to propose strategic locations for new EV charging stations that would best serve the commuting patterns of EV drivers along this route",
        "Based on the traffic volume and patterns of the route, conduct an environmental impact study to assess pollution levels and recommend strategies for reducing emissions",
        "Leverage the insights from the route data to propose a smart city initiative that uses sensors and IoT devices to manage traffic lights and information boards for real-time traffic updates",
        "Write a short story from the perspective of a traveler navigating the given route. Integrate the traffic patterns and weather conditions into the plot, influencing the character's experiences and decisions",
        "Compose a poem that captures the rhythm and flow of the route's traffic. Use the changing weather conditions as metaphors for the emotions and moods encountered along the journey",
        "Write a haiku about this route",
        *[
            f"Write a haiku about this route. It was measured {season}. Add matching kigo"
            for season in [
                "in summer",
                "in winter",
                "in fall",
                "in spring",
                "on new years",
            ]
        ],
        "Write a limerick",
        "Imagine you are a travel blogger documenting a trip along the provided route. Describe the sights, sounds, and atmosphere, incorporating how the traffic ebbs and flows and how the weather shifts throughout your adventure",
        "Set a story on the route in a different historical period. Reflect on how the traffic and weather might have impacted travelers in that era and weave these elements into a narrative that bridges past and present",
        "Create an interactive story or choose-your-own-adventure game where the reader makes decisions based on the data of the route. Their choices should lead to different outcomes and story paths",
        "Author a children's book that personifies vehicles and weather elements along the route. Craft a tale that teaches about traffic safety and the importance of being prepared for weather changes",
        *[
            f"Write a synopsis for a {creator} movie that takes place along this route"
            for creator in [
                "Studio Ghibli",
                "Tarantino",
                "Nolan",
                "Wes Anderson",
                "Sergio Leone",
                "Greta Gerwig",
                "Hitchcock",
                "Tim Burton",
                "Charlie Chaplin",
            ]
        ],
        "Can you recommend me 3 songs that I should listen to on this journey",
        "What is the vibe of this trip",
        "What is the mood of this journey",
        "Can you create a playlist that has the approximately same length as this trip? Also name the songs that you would add so they match the mood",
        "If this trip was a person, what kind of person would it be",
        "Write the lyrics for a song where each verse represents a different segment of the route. Incorporate the emotions and experiences one might encounter due to the traffic and weather changes",
        "If this route was used to create the cover art of an album, which famous album would it be",
        "A band travels along this route. It poses as the inspiration for their next album. Describe what the cover art could look like",
        "If this route was a genre, which would it be and why",
        "Create a buzzfeed quiz. Use this route as inspiration",
        "Create a meme based on this route. Which meme template would you use",
    ]

    NON_TEXT_COLUMNS = []

    def get_system_message(self, style_info=None):
        text, style = super().get_system_message(style_info=style_info)

        return f"{text} Always answer in full sentences.", style
