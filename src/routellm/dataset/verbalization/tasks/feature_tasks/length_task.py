import random

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.verbalization.styles import get_style
from routellm.dataset.verbalization.tasks import BaseTask
from routellm.dataset.verbalization.text_template import splice_route
from routellm.util.geo_conversion import convert_meters_to_length_scale


class LengthTask(BaseTask):
    # Get complete length of route
    NAME = "feature-tasks.length-task"
    VERSION = 0
    TASK_LIST = [
        "How long is this route",
        "What is the length of the following route",
        "Please give me the length of the following route",
        "Can you give me the total length of the route",
        "I was wondering, what is the total length of this route",
        "I'm interested knowing the length of this route",
        "How long would you consider this route",

        "What's the overall distance covered by this route",
        "Can you tell me the entire length of this path",
        "How long is the route in total",
        "What's the sum of all segments in this journey",
        "Could you provide the complete distance for this course",
        "How many miles or kilometers does this whole route span",
        "What is the cumulative length of this travel itinerary",
        "Please let me know how extensive this particular route is",
        "I'd like to find out how much ground we'll cover on this track, can you help with that",
        "Will you please inform me about the aggregate distance along this pathway",
        "I'm interested in determining the comprehensive distance of this route from start to finish; would you mind providing me with that information",
        "As we embark on this journey, it would be helpful to have a clear understanding of the total extent of the route's length; could you please enlighten me about this aspect",
        "In order to better prepare for our trip, I'd like to request your assistance in finding out the complete measurement of the path we'll be following - could you kindly provide that information",
    ]


    def get_system_message(self, style_info=None):
        system_message = """You are an AI assistant that helps people find information about routes.
        
        Keep the language diverse. Make the answer sound how humans would answer to the question. Answer the question, do not describe the route! Give short and precise answers. If not prompted differently, answer in one or 2 sentences!
        
        The answers you give will be used to create an instruction dataset for a multimodal large language model. This model will get special tokens as the routes whereas you get a textual description.
        The textual description is generated with a template and based on feature values from osm, the tomtom traffic api and the azure maps weather api. Use this information as additional knowledge but do not directly include it in your answer.
        Here is a short explanation of every feature in the description:
        
        length: length of the osm segment
        grade: grade in percent calculated from the OSMNX graph and an elevation map
        curvature: menger curvature (1 / curve radius) calculated from osmnx graph
        heading: cardinal direction calculated from the osmnx graph
        landuse: Describes what the area right next to the street is used for. Based on the osm landuse tag but was grouped into new categories
        free flow speed: Free flow speed calculated using the tomtom api
        oneway: Whether the osm segment is a oneway street
        lanes: number of lanes (osm data)
        highway: type of street (converted to text) based on osm street types
        speed limit: based on the osm speed_kph tag
        variable speed: based on the osm maxspeed tag. Describes whether the speed limit on a segment could change based on a sign or a traffic signal.
        bridge: Whether the segment contains a bridge
        tunnel: whether the segment contains a tunnel
        junction: Describes whether this segment is part of a junction/crossing
        traffic incident information: Based on the tomtom traffic api and describes the type of incident, how certain it is that it appears, the magnitude of the incident and the expected delay caused by the incident
        weather information: From the azure maps weather api. Contains precipitation, percentage of cloud coverage, wind speed, wind gust speed and whether there is lightning.
        
        Use the textual description for your answer but do not directly refer to phrasing or the text in your answer. Instead talk about the route/journey/trip/segments/steps/etc.
        """

        if style_info is None:
            style_text, style = self.get_random_style()
        else:
            style = style_info
            style_text = get_style(**style_info)

        return f"{system_message} Follow these phrasing instructions: {style_text}", style # f"{system_message} Follow these style instructions: {style_text}", style

    def generate_task(self, route_df, max_text_length: int = 3.0 * 29000, history=None, style=None, route_path=None):
        # if self.use_llm_for_answering:
        #     warnings.warn("This task does not use an LLM interface (use_llm_for_answering is set to True)!")
        #     self.use_llm_for_answering = False

        cleaned_df = clean_gdf(route_df.copy())

        if cleaned_df is None:
            return None

        splice_list = [list(range(len(cleaned_df)))]
        if len(cleaned_df) > self.MAX_ROUTE_TOKENS and self.allow_splicing:
            splice_list = splice_route(len(cleaned_df), self.MAX_ROUTE_TOKENS)

        task_list = []
        for splice in splice_list:
            splice_df = cleaned_df.iloc[splice]
            system_message, style = self.get_system_message(style_info=style)
            task_text = self.get_task()

            if self.use_llm_for_answering:
                textual_description = self.convert_to_text(splice_df)

                if random.random() < 0.33:
                    task_text = task_text + " Please also provide your reasoning on how you came to your conclusion. "
                message = f"{task_text}"

                message += f"\n{textual_description}"
            else:

                total_length = splice_df["length"].sum()

                length_scale = random.choice(["m", "km", "detailed"])

                distance_info = convert_meters_to_length_scale(total_length, length_scale=length_scale, plural=True)

                if length_scale != "detailed" and random.random() < 0.1:
                    # Add conversion
                    conv_scale = "m" if length_scale == "km" else "km"
                    conv_distance_info = convert_meters_to_length_scale(total_length, length_scale=conv_scale, plural=False)

                    distance_info = f"{distance_info} ({conv_distance_info})"

                message_templates = [
                    f"The total length of the route is {distance_info}.",
                    f"Your route covers a distance of approximately {distance_info}.",
                    f"You'll be traveling a total of {distance_info} on this journey.",
                    f"Expect to cover around {distance_info} throughout your trip.",
                    f"The entire stretch of this route spans about {distance_info}.",
                    f"From start to finish, you'll be traversing {distance_info}.",
                    f"The comprehensive distance for this path measures at {distance_info}.",
                    f"This particular route encompasses a total of {distance_info}.",
                    f"Your expedition will span a distance of roughly {distance_info}.",
                    f"While following this track, you will cover an estimated {distance_info}.",
                    f"Embarking on this adventure, you'll find yourself covering an impressive {distance_info} in total.",
                    f"As your journey unfolds across the vast landscape, the cumulative distance stretches out to {distance_info}.",
                    f"From its beginning to its end, this intriguing expedition encompasses a remarkable {distance_info}.",
                    f"Like a winding path through time and space, your route will ultimately span a noteworthy {distance_info}.",
                    f"The odyssey that awaits you is set to unfurl over an extensive course of approximately {distance_info}.",
                    f"As you traverse this expansive terrain, prepare for a voyage spanning an impressive distance of {distance_info}.",
                    f"In the grand scheme of your trek, the entire length comes together as a majestic sum of around {distance_info}.",
                    f"With each step forward and every mile conquered along this route, you're on an amazing journey totaling {distance_info}.",
                    f"A venture stretching across diverse landscapes and experiences brings forth a fascinating total distance of roughly {distance_info}.",
                    f"Traveling from one point to another on this unique path creates a memorable passage measuring about {distance_info}.",
                    f"As you embark on this captivating journey through a diverse array of landscapes, taking in the sights and sounds along the way, it's worth noting that your adventure encompasses an awe-inspiring {distance_info}, showcasing just how vast and remarkable this planet truly is.",
                    f"Throughout every twist and turn of this epic voyage, with each new experience blending into the next like chapters in a riveting novel, it's astonishing to think that you'll be covering a staggering {distance_info} during your travels - a testament to the incredible undertaking ahead.",
                    f"Your venture shall unfold like an elaborate tapestry woven from threads of countless memories and discoveries, which together weave a resplendent tableau spanning across the phenomenal distance of {distance_info}; it is within this grand narrative that you will uncover hidden treasures and forge unforgettable moments.",
                    f"Ah, good question! The entire distance you'll be covering on this journey is approximately {distance_info}.",
                    f"Well, now that you ask, your trip will span a distance of roughly {distance_info}.",
                    f"Gotcha! In response to your query, the whole route stretches for about {distance_info}.",
                    f"I see what you're asking - your adventure actually encompasses nearly {distance_info} in total.",
                    f"That's a great question! It turns out that this path measures around {distance_info} from start to finish.",
                    f"To answer your question, the complete distance of the planned route is close to {distance_info}.",
                    f"Sure thing! In response to your inquiry, the overall length of this journey comes out to be roughly {distance_info}.",
                    f"To provide some clarity regarding your question: the full extent of this trek spans about {distance_info}.",
                ]
                system_message += f"Original task: {task_text}."
                message = random.choice(message_templates)

            task_description = {
                "system": system_message,
                "message": message,
                "style": style,
                "segments": None,
                "header_text": None,
                "address_header_text": None,
                "segment_texts": None,
                "task_text": task_text,
                "include_header": False,
                "print_step_id": False,
                "use_orig_steps": False,
                "splice_indices": splice_df["step"].to_list(),
                "splice_arrays": True,
                "raw_route": None,
                "task_name": self.NAME,
            }
            task_list.append(self.process_task(task_description, history=history))

        return task_list
