"""Module for converting constants to textual representations."""

HIGHWAY_TO_TEXT = {
    "unclassified": "unclassified road type",
    "secondary": "secondary road",
    "tertiary": "tertiary road",
    "residential": "residential road",
    "primary": "primary road",
    "primary_link": "link to/from a primary road",
    "secondary_link": "link to/from a secondary road",
    "trunk": "trunk road",
    "living_street": "living street",
    "trunk_link": "link to/from a trunk road",
    "tertiary_link": "link to/from a tertiary road",
    "motorway_link": "link to/from a motorway",
    "motorway": "motorway",
    "busway": "busway",
    "disused": "disused road",
    "rest_area": "rest area",
    "emergency_bay": "emergency bay",
    "road": "road",
}


def convert_highway_to_text(
    value: str,
    plural: bool = False,
    article: bool = False,
    definite_article: bool = False,
) -> str:
    """
    Convert a highway type to a human-readable text representation.

    Args:
        value (str): Highway type to convert.
        plural (bool, optional): If True, the description should be in plural. \
            Defaults to False.
        article (bool, optional): If True, add an article to the description. \
            Defaults to False.
        definite_article (bool, optional): If False, use an indefinite article. \
            Defaults to False.

    Returns:
        str: Human-readable text representation of the highway type.

    """
    assert not plural or not article or definite_article, (
        "Cannot use indefinite article and plural form!"
    )
    if not plural:
        selected_article = (
            "a " if article and not definite_article else "the " if article else ""
        )
        selected_article_vowel = (
            "an " if article and not definite_article else "the " if article else ""
        )
        highway_map = {
            "unclassified": f"{selected_article_vowel}unclassified road",
            "secondary": f"{selected_article}secondary road",
            "tertiary": f"{selected_article}tertiary road",
            "residential": f"{selected_article}residential road",
            "primary": f"{selected_article}primary road",
            "primary_link": f"{selected_article}link to/from a primary road",
            "secondary_link": f"{selected_article}link to/from a secondary road",
            "trunk": f"{selected_article}trunk road",
            "living_street": f"{selected_article}living street",
            "trunk_link": f"{selected_article}link to/from a trunk road",
            "tertiary_link": f"{selected_article}link to/from a tertiary road",
            "motorway_link": f"{selected_article}link to/from a motorway",
            "motorway": f"{selected_article}motorway",
            "busway": f"{selected_article}busway",
            "disused": f"{selected_article}disused road",
            "rest_area": f"{selected_article}rest area",
            "emergency_bay": f"{selected_article}emergency bay",
            "road": f"{selected_article}road",
        }
    else:
        selected_article = 'the ' if article else ''
        highway_map = {
            "unclassified": f"{selected_article}unclassified roads",
            "secondary": f"{selected_article}secondary roads",
            "tertiary": f"{selected_article}tertiary roads",
            "residential": f"{selected_article}residential roads",
            "primary": f"{selected_article}primary roads",
            "primary_link": f"{selected_article}link to/from a primary roads",
            "secondary_link": f"{selected_article}link to/from a secondary roads",
            "trunk": f"{selected_article}trunk roads",
            "living_street": f"{selected_article}living streets",
            "trunk_link": f"{selected_article}links to/from a trunk road",
            "tertiary_link": f"{selected_article}links to/from a tertiary road",
            "motorway_link": f"{selected_article}links to/from a motorway",
            "motorway": f"{selected_article}motorways",
            "busway": f"{selected_article}busways",
            "disused": f"{selected_article}disused roads",
            "rest_area": f"{selected_article}rest areas",
            "emergency_bay": f"{selected_article}emergency bays",
            "road": f"{selected_article}roads",
        }

    return highway_map[value]


INCIDENT_TO_TEXT = {
    "Unknown": "no incidents",
    "Cluster": "no incidents",
    "Accident": "an accident",
    "Fog": "heavy fog",
    "Dangerous Conditions": "dangerous driving conditions",
    "Rain": "heavy rain",
    "Ice": "ice",
    "Jam": "traffic jam",
    "Lane Closed": "a closed lane",
    "Road Closed": "a closed road",
    "Road Works": "road works",
    "Wind": "strong wind",
    "Flooding": "a flooding",
    "Broken Down Vehicle": "a broken down vehicle",
}

INCIDENT_DESCRIPTIONS = {
    # "Unknown": [],
    "Accident": [
        "An accident category represents a situation where one or more vehicles have collided, resulting in potential traffic congestion and delays. This can include minor fender benders or major crashes, and might involve additional emergency services.",
        "A sudden and unexpected event on the road involving one or more vehicles that can lead to traffic congestion, property damage, or personal injury, often requiring emergency services to respond.",
        "An unfortunate vehicular incident that disrupts normal traffic flow, potentially causing injuries and damage to the surroundings, and may require a coordinated response from emergency personnel to manage the situation.",
        "An accident occurs when vehicles collide or crash into objects, causing traffic delays, potential injuries, and sometimes requiring emergency services to help resolve the situation and clear the road for other drivers.",
        "An accident constitutes an unanticipated vehicular collision or impact with stationary objects, often resulting in extensive property damage, potential injuries, and traffic congestion; such incidents necessitate a coordinated response from emergency personnel to manage recovery efforts and restore normalcy.",
    ],
    "Fog": [
        "A meteorological phenomenon characterized by suspended water droplets in the air leading to reduced visibility for drivers, which can increase the risk of accidents due to impaired line of sight and slower reaction times.",
        "A dense cloud of moisture particles near the ground level that obstructs drivers' vision and makes navigating roads difficult, necessitating greater caution and vigilance while driving to avoid potential hazards.",
        "Fog is a weather condition where water droplets in the air create a thick mist that reduces visibility for drivers, making it crucial to drive slowly, use headlights appropriately, and maintain a safe distance from other vehicles on the road.",
        "Fog is a meteorological phenomenon characterized by the presence of suspended water droplets or ice crystals in the air near ground level, leading to reduced visibility for drivers and thus increasing the risk of accidents due to impaired sightlines and a higher likelihood of human error.",
    ],
    "Dangerous Conditions": [
        "Situations on the road that pose immediate threats to drivers and pedestrians due to hazardous elements such as debris, oil spills, or loose gravel; these conditions often require swift action from authorities to mitigate risks and ensure safety.",
        "Potentially dangerous circumstances arising from various factors including environmental elements or human intervention that can increase the likelihood of accidents on roadways; awareness and cautious navigation is crucial for avoiding incidents in such conditions.",
        "A dangerous condition refers to any situation on the road that poses a risk to drivers and passengers, such as unexpected obstacles, slippery surfaces due to oil spills or loose gravel, or poor infrastructure that requires extra caution and attention while driving.",
        "A dangerous condition refers to any hazardous circumstance on roadways arising from various factors including environmental elements, poor infrastructure maintenance, or unforeseen obstacles; such conditions require heightened awareness from drivers as well as swift action from relevant authorities to mitigate risks and ensure safety.",
    ],
    "Rain": [
        "An atmospheric condition where heavy precipitation creates wet surfaces making roads slippery with reduced visibility for drivers, potentially increasing the risk of hydroplaning hazards or other accidents if proper precautions are not taken.",
        "Persistent showers leading to poor driving conditions as falling water droplets make it difficult for motorists to see clearly while also reducing grip on road surfaces; this requires extra care when driving as stopping distances may be increased.",
        "Rainy weather can lead to wet roads with reduced visibility for drivers; this increases the risk of accidents and makes it essential for motorists to exercise caution by slowing down, using their windshield wipers effectively, and maintaining sufficient distance from other vehicles.",
        "Rain-induced driving conditions involve increased risks due to wet surfaces that reduce traction for vehicles while simultaneously impairing visibility through the accumulation of water droplets on windshields; prudent driving techniques should be employed during rainfall events to avoid hydroplaning hazards and maintain safe stopping distances.",
    ],
    "Ice": [
        "The presence of frozen water on road surfaces due to cold temperatures increases the risk of skidding or loss of vehicle control during travel, necessitating careful navigation and appropriate tire selection during winter months.",
        "Hazardous driving conditions caused by subzero temperatures resulting in ice formation on streets; glazed surfaces can make braking difficult for vehicles leading to a heightened possibility of collisions if extra caution is not exercised.",
        "Icy roads present hazardous conditions for drivers as ice-covered surfaces can cause vehicles to skid or lose control; during winter months or in areas prone to frost formation, it is important to drive cautiously and use appropriate tires for better traction.",
        "Ice formation on roadways presents a significant hazard for motorists as friction between tires and pavement is greatly reduced, increasing the risk of skidding and loss of control; appropriate precautionary measures should be taken during icy weather conditions including utilizing winter tires, maintaining slower speeds, and avoiding sudden braking maneuvers.",
    ],
    "Jam": [
        "A situation where excessive demand exceeds roadway capacity causing a significant slowdown in traffic flow or extended waiting periods for commuters, often attributed to peak hours or large events attracting numerous vehicles.",
        "Traffic congestion caused by an accumulation of cars hindering mobility with several contributing factors such as bottlenecks at merging points or too many vehicles attempting entry into limited available lanes at once.",
        "Traffic jams occur when there are too many cars on the road at once or due to incidents like accidents or construction projects; these situations can lead to long waiting times for commuters and require patience as well as alternative route planning if possible.",
        "Traffic congestion manifests when roadway capacity is exceeded by vehicle demand leading to hindered mobility, longer travel times, and increased fuel consumption; this can be attributed to multiple factors such as infrastructure limitations, peak hour influxes, or bottlenecks at merging points.",
    ],
    "Lane Closed": [
        "The temporary unavailability or blocking off of a specific lane due to construction work, maintenance activities, accident response efforts or other safety measures aimed at protecting both workers and motorists during active work zones.",
        "Partial closure of one or more lanes on a roadway typically indicated by cones or barriers directing traffic around the affected area; this can result in reduced available space for driving requiring an adaptation in motorists' behavior such as merging into alternate lanes.",
        "When a lane is closed on a road due to ongoing construction work, maintenance activities or accident response efforts, drivers must adapt by merging into alternate lanes and paying close attention to signs directing them around the affected area.",
        "A lane closure occurs when one or more lanes on a roadway are temporarily unavailable due to ongoing construction activities, maintenance operations or incident response efforts; this necessitates careful navigation by drivers who must adapt their behavior accordingly by merging into alternate lanes and following posted signage.",
    ],
    "Road Closed": [
        "Complete shutdown of a street or highway due either ongoing construction projects, maintenance work, emergency situations or other safety concerns that necessitate detouring traffic to alternative routes until the road can be reopened.",
        "The closure of an entire roadway for a specific period of time, usually marked by barricades and signage indicating no passage is allowed for vehicular traffic; drivers must seek alternate routes and follow detour information provided by authorities.",
        "A road closure signifies that an entire street or highway has been shut down temporarily due to reasons such as construction projects, maintenance work, emergencies or safety concerns; during this time, motorists must follow detour signs provided by authorities in order to navigate around the closure.",
        "Road closures entail the temporary shutdown of an entire street or highway segment due either to construction projects (e.g., repaving), maintenance work (e.g., utility installations), emergency situations (e.g., natural disasters) or other safety concerns that require vehicular traffic to seek alternate routes until reopening occurs.",
    ],
    "Road Works": [
        "Ongoing engineering projects involving construction activities such as repaving, widening or building new infrastructure on roads aimed at improving transportation systems; these works can cause temporary alterations to normal traffic flow and may require additional caution from drivers.",
        "Planned maintenance operations like pothole repair, drainage installation or utility work that temporarily disrupt regular traffic conditions in order to enhance existing roadway systems including pavement markings, signage, signals, and overall safety.",
        "Road works involve various construction activities aimed at improving transportation infrastructure like widening roads or fixing potholes; these projects may temporarily disrupt normal traffic flow but ultimately result in safer and more efficient travel conditions once completed.",
        "Road works encompass diverse engineering projects aimed at enhancing transportation systems such as repaving initiatives, infrastructure widening undertakings or new developments; these activities may temporarily disrupt regular traffic flow but are essential for optimizing safety standards and accommodating growth within transportation networks.",
    ],
    "Wind": [
        "Strong gusts of wind causing reduced visibility for drivers due to blowing dust or debris while also affecting vehicle stability during travel; this requires increased caution as steering may become difficult and lighter vehicles are more susceptible to being blown off course.",
        "High-velocity air currents that can impact the control of a vehicle on the road during windy weather conditions; drivers should be extra vigilant when navigating through areas prone to strong winds and adjust their speed accordingly to maintain safety.",
        "Strong winds can impact driving conditions by reducing visibility due to blowing dust or debris and affecting vehicle stability; under such circumstances, it's vital for drivers to be cautious and vigilant while navigating through windy areas.",
        "High-wind events pose challenges for motorists through reduced visibility caused by blowing dust or debris along with potential impacts on vehicle stability particularly for high-profile vehicles like trucks or buses; driving practices should be adapted accordingly during windy conditions with greater caution exercised to maintain safety.",
    ],
    "Flooding": [
        "Excessive water accumulation on streets due to heavy rainfall or overflowing bodies of water which can lead to impassable roads with potential hydroplaning risks for vehicles attempting traversal through flooded areas.",
        "Rapidly rising water levels resulting in submerged roads obstructing safe passage and posing dangers such as hidden debris, depth-related hazards, or vehicle damage if attempted traversal occurs without proper caution.",
        "Flooding occurs when heavy rainfall causes water levels in streets or highways rise rapidly leading submerged roads presenting hazards like hidden debris; during floods driving should be avoided whenever possible but if necessary extreme care should be taken",
        "Flooding-related disruptions occur when excessive water accumulation engulfs roadways, obstructing safe passage and presenting dangers including submerged debris, depth-related hazards, or vehicle damage; during flood events, authorities may issue travel advisories to avoid unnecessary driving or to exercise extreme caution if travel is unavoidable.",
    ],
    "Broken Down Vehicle": [
        "A disabled car on the side of the road impeding traffic flow due to mechanical failure or other issues requiring assistance from roadside services; this creates congestion as other drivers slow down or change lanes to avoid the stalled vehicle.",
        "An immobile truck or bus causing obstruction while waiting for help on a busy roadway; stranded motorists create delays for others as they await necessary repairs or removal from main thoroughfares by tow trucks or emergency personnel.",
        "A broken-down vehicle typically refers situations where cars become immobile roadside issues mechanical failure others slow down move past stalled car safely possible considerate driver's predicament",
        "A broken-down vehicle refers to a stranded motorist who is rendered immobile due to mechanical failures or technical difficulties; this can impede traffic flow as other drivers are required to slow down or change lanes in order to safely navigate around the disabled vehicle while awaiting assistance from roadside services.",
    ],
}


def convert_incident_to_text(
    value: str,
    plural: bool = False,
    article: bool = False,
    definite_article: bool = False,
) -> str:
    """
    Convert an incident type to a human-readable text representation.

    Args:
        value (str): Incident type to convert.
        plural (bool, optional): If True, the description should be in plural. \
            Defaults to False.
        article (bool, optional): If True, add an article to the description. \
            Defaults to False.
        definite_article (bool, optional): If False, use an indefinite article. \
            Defaults to False.

    Returns:
        str: Human-readable text representation of the incident type.

    """
    assert not plural or not article or definite_article, (
        "Cannot use indefinite article and plural form!"
    )
    article_text = "the " if article and definite_article else "a " if article else ""
    vowel_article_text = (
        "the " if article and definite_article else "an " if article else ""
    )
    plural_text = "s" if plural else ""

    text_map = {
        "Unknown": f"no incident{plural_text}",
        "Cluster": f"no incident{plural_text}",
        "Accident": f"{vowel_article_text}accident{plural_text}",
        "Fog": f"{vowel_article_text}instance{plural_text} of heavy fog",
        "Dangerous Conditions": f"{article_text}dangerous driving condition{plural_text}",
        "Rain": f"{vowel_article_text}instance{plural_text} of heavy rain",
        "Ice": f"{vowel_article_text}instance{plural_text} of ice",
        "Jam": f"{article_text}traffic jam{plural_text}",
        "Lane Closed": f"{article_text}closed lane{plural_text}",
        "Road Closed": f"{article_text}closed road{plural_text}",
        "Road Works": f"{article_text}construction site{plural_text}",
        "Wind": f"{article_text}strong wind{plural_text}",
        "Flooding": f"{article_text}flooding{plural_text}",
        "Broken Down Vehicle": f"{article_text}broken down vehicle{plural_text}",
        "Cluster": f"{article_text}incident cluster{plural_text}",
    }

    return text_map[value]


PLURAL_INCIDENT_TEXTS = [
    "Unknown",
    "Cluster",
    "Dangerous Conditions",
    "Road Works",
]


def convert_landuse_to_text(
    value: str,
    plural: bool = False,
    article: bool = False,
    definite_article: bool = False,
) -> str:
    """
    Convert a land use type to a human-readable text representation.

    Args:
        value (str): Land use type to convert.
        plural (bool, optional): If True, the description should be in plural. \
            Defaults to False.
        article (bool, optional): If True, add an article to the description. \
            Defaults to False.
        definite_article (bool, optional): If False, use an indefinite article. \
            Defaults to False.

    Returns:
        str: Human-readable text representation of the land use type.

    """
    assert not plural or not article or definite_article, (
        "Cannot use indefinite article and plural form!"
    )
    article_text = "the " if article and definite_article else "a " if article else ""
    vowel_article_text = (
        "the " if article and definite_article else "an " if article else ""
    )
    plural_text = "s" if plural else ""

    text_map = {
        "residential": f"{article_text}residential area{plural_text}",
        "commercial": f"{article_text}commercial area{plural_text}",
        "agriculture": f"{vowel_article_text}area{plural_text} for agricultural usage",
        "institutional": f"{vowel_article_text}area{plural_text} with institutional buildings",
        "waterbody": f"{vowel_article_text}area{plural_text} containing bodies of water",
        "greenery": f"{article_text}green or recreational area{plural_text}",
        "construction": f"{article_text}construction site{plural_text}",
        "forest": f"{article_text}forest{plural_text}",
        "industrial": f"{vowel_article_text}industrial area{plural_text}",
        "transporation": f"{article_text}area{plural_text} used for transportation",
        "cemetery": f"{article_text}{'cemeteries' if plural else 'cemetery'}",
        "other": f"{vowel_article_text}area{plural_text} for miscellaneous usage",
    }

    return text_map[value]


LANDUSE_DESCRIPTION = {
    "other": [
        "This area includes land used for various purposes such as conservation, landfill, military, winter sports, user defined, civic admin, proposed, and traffic island. These are areas that do not fit neatly into other categories and serve a wide range of functions.",
        "This area includes various land uses that don't fit into other specific categories. It encompasses a diverse range of purposes and activities.",
        "This type of area includes various land uses that don't fit into other specific categories, such as military zones, winter sports facilities, traffic islands, conservation lands, proposed developments, and civic administration areas.",
        "This type of area is used for a variety of purposes that don't fit into other categories. It can include military bases, traffic islands, winter sports facilities, and proposed land developments. These areas might be frequented by soldiers, athletes, or construction workers depending on the specific use.",
        "This area is used for various purposes that don't fit into specific categories. It may include military zones, landfills, conservation areas, traffic islands, winter sports locations, and proposed construction sites. People from different backgrounds and professions might visit these places depending on their purpose. They may not be perceived as beautiful compared to other areas, as their key features vary widely.",
    ],
    "commercial": [
        "Commercial areas consist of land designated for commercial activities such as businesses, shops, and services. This includes commercial and fairground areas as well as retail spaces.",
        "Commercial areas include land used for activities such as commerce and trade. In this category, you will find land designated as commercial, fairground, and retail. These areas typically contain businesses like shops, offices, and entertainment venues.",
        "A commercial area comprises land used for business-related activities like retail stores and shopping centers, offices, or service establishments. In the given dictionary, 'commercial', 'fairground', and 'retail' fall under this category.",
        "Commercial areas are places where business activities take place, such as shopping centers or office buildings. These also include retail spaces and fairgrounds which host events like trade shows or exhibitions.",
        "Commercial areas are places where businesses operate and sell goods or services to customers. They usually include shops, offices, restaurants, and hotels. People who work or visit these areas are typically business owners, employees, shoppers, and tourists.",
        "Commercial areas are places where businesses and commercial activities take place. They include shops, offices, restaurants, and other retail establishments. These areas are frequented by people who work or shop in the businesses located there. The key feature of commercial areas is that they generate economic activity. Their beauty can vary depending on the architecture and design of the buildings and public spaces.",
    ],
    "construction": [
        "Construction sites are areas where new buildings or infrastructure are being built or existing structures are undergoing significant renovations. These include construction and brownfield sites.",
        "Construction sites are areas where new buildings or infrastructure are being built or renovated. This category groups together construction and brownfield land uses. Brownfields are previously developed lands that may require environmental cleanup before they can be redeveloped.",
        "Construction sites are areas where new buildings or infrastructure are being built. This can include brownfields, which are previously developed lands that are being redeveloped for new purposes.",
        "A construction site is an area where buildings or infrastructure are being built or renovated. These sites may contain heavy machinery and materials needed for construction work. Construction sites are mainly used by builders, architects, engineers, and other professionals working in the construction industry.",
        "Construction sites are where new buildings or infrastructure are being built or existing ones are being renovated. They can include brownfield land which is previously developed land now earmarked for new construction projects. These places are visited by construction workers and professionals involved in building projects. Construction sites generally aren't considered beautiful due to the presence of heavy machinery and ongoing work but can represent progress and development.",
    ],
    "institutional": [
        "Areas with institutional buildings are locations designated for facilities that provide public services such as education, healthcare, religious activities or government administration. These include education institutions like schools or universities; hospitals under healthcare; churches or temples under religious; and civic_admin buildings.",
        "Areas with institutional buildings are designated for public or private organizations that provide various services to the community. These areas include education (schools and universities), institutional (government buildings), religious (churches and mosques), and healthcare facilities (hospitals and clinics).",
        "Areas with institutional buildings include places like schools or universities (education), hospitals (healthcare), religious institutions like churches or mosques, and government offices.",
        "An area with institutional buildings contains facilities like schools, hospitals, religious institutions (churches/temples), and government offices. This type of area is frequented by students, teachers, doctors, nurses, religious leaders as well as worshippers attending services or ceremonies.",
        "Areas with institutional buildings serve a variety of public needs such as education, religious activities, healthcare facilities, and civic administration purposes. These places might be visited by students, patients, religious followers or government employees among others. Key features include large structures designed to accommodate many people or provide services to the community. The perceived beauty of these areas depends on architectural style and maintenance of the facilities.",
    ],
    "industrial": [
        "Industrial areas comprise lands designated for manufacturing and processing activities including factories and warehouses. This includes industrial zones as well as quarries which serve an industrial function by extracting raw materials.",
        "Industrial areas are used for manufacturing goods or processing raw materials. This category encompasses both industrial zones and quarries where extraction of natural resources occurs.",
        "Industrial areas are places where manufacturing, production, or extraction of resources takes place. These can include factories, warehouses, and quarries.",
        "Industrial areas are places where manufacturing processes take place or raw materials are extracted from the earth (e.g., quarries). These areas may have factories or warehouses where products are made or stored. Workers in industrial areas include factory workers and truck drivers transporting goods.",
        "Industrial areas contain factories, manufacturing plants, warehouses, or quarries where goods are produced or processed. Workers in these industries use this space for their daily tasks while some visitors might come for business-related purposes. Key features include large-scale industrial structures like warehouses or factories that serve a specific purpose in producing goods or services. Industrial areas usually aren't considered beautiful due to the focus on functionality over aesthetics.",
    ],
    "residential": [
        "Residential areas are locations designated primarily for housing purposes such as single-family homes or apartment complexes. This includes residential zones where people live in houses or apartments on a permanent basis.",
        "Residential areas are primarily used for housing purposes. They contain different types of dwellings such as single-family homes, apartments, or condominiums.",
        "Residential areas consist of homes and living spaces for people. These can include houses, apartments, condos, and other types of housing units.",
        "Residential areas contain houses and apartments where people live with their families. These neighborhoods often have parks and playgrounds for recreation purposes alongside local shops catering to daily needs. People using residential areas primarily include residents along with delivery personnel supplying groceries/services.",
        "Residential areas consist mainly of houses or apartments where people live with their families or alone. They often have parks, schools nearby serving as social spaces for residents to interact with each other peacefully—these neighborhoods vary in size (from small towns to cities). This space's key features include homes surrounded by streets designed with safety considerations; landscaping enhances residents' quality of life through beautification efforts around properties (e.g., gardens).",
    ],
    "agriculture": [
        "Areas for agricultural use include lands dedicated to farming practices such as growing crops or raising livestock. These encompass aquaculture; allotments; farmland; farmyard; paddy fields used to cultivate rice; animal keeping areas; greenhouse horticulture facilities; orchards where fruit trees grow; plant nurseries that propagate plants for sale; and vineyards producing grapes for wine production.",
        "Agricultural areas are dedicated to farming activities including crop cultivation and animal husbandry. This category covers aquaculture (fish farming), allotments (small plots of land allocated to individual gardeners), farmland, farmyard (areas around farm buildings), paddy fields (for rice cultivation), animal keeping facilities, flowerbeds used in agriculture settings, greenhouse horticulture operations, orchards where fruit trees grow in a concentrated fashion , plant nurseries propagating plants for sale , vineyards producing grapes",
        "Agricultural use areas refer to lands used for farming practices like growing crops or raising livestock. Examples include aquaculture (fish farming), farmland for crops, farmyards for animals keeping and equipment storage, greenhouses for plant cultivation in controlled environments, paddy fields for rice cultivation orchards for fruit trees planting , vineyards for grape growing and plant nurseries specializing in the growth of young plants.",
        "Areas for agricultural use consist of farmland like orchards and vineyards as well as aquaculture facilities raising fish/shellfish in artificial ponds/tanks. Farmers grow crops/raise animals here to produce food products that eventually reach our tables. Farmworkers tend crops/animals while truck drivers transport these products to markets/distribution centers.",
        "Agricultural areas are lands used primarily for farming crops like fruits vegetables grains also livestock animal keeping aquaculture (fish farming), etc.—all aimed towards producing food resources humans consume daily basis farmworkers frequent farmers market vendors sell fresh produce grown within these boundaries you'd find fields greenhouses orchards vineyards nursery plantations nurseries grazing pastureland Key characteristic extensive open predominantly rural scenic picturesque beautiful settings varying upon cultivated landscape condition.",
    ],
    "greenery": [
        "Green or recreational areas consist of open spaces that provide opportunities for relaxation and leisure activities such as parks, playgrounds or sports fields. These include allotments which can be used both recreationally and agriculturally; flowerbeds serving decorative purposes in urban environments; grassy expanses often found in public parks and green spaces; greenfield lands which are undeveloped open spaces typically within city boundaries used recreationally but not intensively developed yet.; recreation grounds designed specifically to host sporting events or outdoor games like soccer fields.; village greens traditionally located at the center of rural settlements providing informal recreational space within communities.",
        "Green or recreational areas provide space for leisure activities or natural environments within urban settings . Commonly included within this classification are communal gardens, publicly managed flowerbeds, meadows, landscaped lawns, greenfields, recreation grounds and traditional open space in the center of settlements.",
        "Green or recreational areas are meant to be enjoyed by the public and can range from parks to gardens. They may feature grassy fields, flowerbeds with colorful blossoms, meadows with wildflowers, allotments allowing people to grow their own plants and recreation grounds providing outdoor activities.",
        "Green or recreational areas include parks/playgrounds that provide space for leisure activities like picnicking/jogging/playing sports etc., allotments where individuals can grow plants privately & meadowlands which serve as an important habitat for wildlife/fauna providing ecosystem benefits. Families/people looking to relax engage in leisure activities frequent these spaces.",
        "Green recreational spaces designed promote leisure relaxation enjoyment among communities—often encompassing parks gardens playgrounds sports fields allotments meadows village greens grassy lawns They're ideal spots picnics walks games exercise many activities residents tourists alike benefit mentally physically from spending time surrounded by nature lush greenery peaceful atmosphere adds overall beauty area.",
    ],
    "forest": [
        "Forests consist of large tracts of wooded land covered predominantly by trees.",
        "Forest regions consist primarily of densely wooded landscapes filled with trees. They might be managed forests intended for logging, biodiversity conservation efforts or recreational pursuits.",
        "Forests are large natural wooded environments containing a diverse mix of trees species covering a wide expanse. They provide habitat to various wildlife species, support biodiversity, serve as carbon sinks thus playing an important role in regulating climate change.",
        "Forests are large areas of land covered primarily by trees and other vegetation. They provide a habitat for various plants and animals, maintain ecological balance, and act as carbon sinks. Forests can be used by people for recreational activities like hiking or bird-watching, or for sustainable logging of wood resources.",
        "Forests are large areas covered primarily by trees and other vegetation. They provide habitat for wildlife while serving as natural resources (timber) helping regulate climate balance carbon dioxide levels key features dense plant life canopies trails camping spots sometimes streams rivers People frequent forests variety reasons including hiking bird-watching studying plants animals Researchers students nature enthusiasts enjoy beauty serenity offered forested landscapes.",
    ],
    "waterbody": [
        "Area containing bodies of water refers to any region wherein water is its predominant feature like water basins, reservoirs, salt ponds, ports, or harbours. These can be man-made or natural features which include but are not limited to lakes, rivers and other bodies of water used for various purposes such as recreation, transportation or resource extraction.",
        '"Area containing bodies of water" refers to locations which have man-made features designed specifically to hold water. Examples contained within this designation include basins utilized to control floodwaters; reservoirs created through the damming rivers; salt ponds constructed harvesting salt from seawater; harbors offering sheltered harborages ships',
        "Areas containing bodies of water refers to geographical locations dominated by water features such as lakes, rivers, reservoirs  which stores water supplies ,harbours providing shelter to boats from rough waters waves, salt ponds utilized in producing salts from evaporation processes & basins acting as catchment systems during heavy rainfalls.",
        "Areas containing bodies of water include basins, reservoirs, salt ponds, harbors, and ports. These places can be used for activities such as fishing, boating, and transportation of goods via ships. People who use these areas might be fishermen, sailors, dockworkers or tourists enjoying water-based activities.",
        "Areas containing bodies of water include natural or man-made features like lakes, rivers, reservoirs, salt ponds, harbors, and ports. They serve various purposes such as providing water for drinking, irrigation, transportation routes, or recreational activities like swimming and boating. People who visit these areas might include workers in the transportation or water industries, locals who enjoy water-based activities or those who simply appreciate the scenic beauty provided by these bodies of water.",
    ],
    "cemetery": [
        "A cemetery is a designated area where the remains of deceased individuals are buried or otherwise interred.",
        "Cemeteries encompass plots land specifically designated as burial grounds . This could include traditional cemeteries, memorial parks , graveyards attached church property",
        "Cemeteries are dedicated spaces for burial of deceased individuals, and they often contain graves, tombstones, or other monuments to commemorate the lives of the departed. They may be associated with religious institutions or simply serve as a public space for honoring loved ones.",
        "A cemetery is a place where deceased individuals are buried or their ashes interred. Cemeteries are often visited by people mourning the loss of loved ones or paying respects to ancestors. In addition to providing a resting place for the dead, cemeteries can also have historical or cultural significance.",
        "A cemetery is a designated area where people are buried after they pass away. It often features tombstones or monuments to honor the deceased and serves as a place for families and friends to remember their loved ones. Cemeteries are typically well-maintained with green spaces and pathways for visitors to walk through. Though some may find cemeteries somber due to their association with death, others may perceive them as tranquil spaces that offer quiet reflection.",
    ],
    "transporation": [
        "Areas used for transportation are locations that facilitate the movement of people and goods through various modes like roads, railways, depots or garages. These include depots serving as facilities to support transport infrastructure; garages accommodating parking needs; railway lands dedicated for train tracks and related infrastructure.",
        "Areas used for transportation are spaces dedicated to the movement of people or goods, including depots (places where vehicles or equipment are stored), garages (parking areas for cars and other types of vehicles), railway tracks (used by trains) and surrounding areas.",
        "Transportation areas includes locations designated specially for facilitating movement & travel on different modes - cars trucks trains etc., It could be depots serving as storage facility point goods & vehicles along transportation networks garages housing private vehicles railways supporting train travel all around countries cities.",
        "An area used for transportation includes infrastructure such as depots (for storing vehicles), garages (for parking/maintaining vehicles), railways (for train transport), and port facilities (for shipping). People using these areas include commuters travelling to work/school/home; drivers/operators maintaining vehicles & equipment; and workers involved in logistics/transportation industries.",
        "Areas used for transportation include places where various modes of transport operate, such as roads, railways, depots, and garages. These areas facilitate the movement of people and goods from one location to another. They are frequented by travelers, commuters, transport workers like bus drivers or train operators, and others involved in the transportation industry. Key features include infrastructure like tracks, stations or platforms, and parking spaces. While these areas might not be considered beautiful in a traditional sense due to their utilitarian nature, they play a vital role in connecting communities and enabling economic activities.",
    ],
}

MAGNITUDE_DESCRIPTIONS = {
    "unknown": [
        "For the driver, incidents in this category will likely have minimal to no effect on their journey. They might observe an event taking place by the roadside, but it won't cause any significant delays or require them to change their planned route.",
        "No or unknown Impact: Incidents in this category have minimal to no effect on drivers and passengers, as they don't cause significant delays or require route changes. Drivers might feel a sense of relief that the incident hasn't affected their journey, while passengers remain relaxed and unbothered by the situation.",
        "For the driver, incidents in this category will likely have minimal to no effect on their journey. They might observe an event taking place by the roadside, but it won't cause any significant delays or require them to change their planned route.",
        "This category refers to incidents where either the exact impact on traffic is yet to be determined, or there is no significant disruption caused. These could include minor roadworks, unverified accidents, or other minor events that do not noticeably affect the flow of traffic.",
        "This category is assigned to incidents when there is not enough information to determine the impact or cause accurately. It could be due to a sudden change in traffic flow or an unreported event affecting the area. For traffic, this means there may be potential disruptions or delays that cannot be predicted.",
    ],
    "minor": [
        "When facing a minor impact incident, drivers may experience a slight increase in travel time due to slowed traffic or a temporary reduction of lanes. These situations are typically short-lived and don't require drivers to take alternative routes or significantly adjust their plans.",
        "Encountering minor impact incidents, such as temporary lane closures due to maintenance work, results in slight increases in travel time for drivers and passengers. They may experience mild frustration or impatience but generally feel understanding since these disruptions are short-lived and don't require major alterations to their planned route.",
        "Minor Impact: In this category, incidents cause a slight disruption in the normal flow of traffic but do not create severe delays or problems for commuters. Examples may include small accidents quickly cleared by emergency services, temporary lane closures due to maintenance work, or partially blocked roads due to debris.",
        "When facing a minor impact incident, drivers may experience a slight increase in travel time due to slowed traffic or a temporary reduction of lanes. These situations are typically short-lived and don't require drivers to take alternative routes or significantly adjust their plans.",
        "Minor incidents usually involve minor accidents, roadwork, or other small-scale events that have limited impact on the overall traffic flow. These incidents can cause slight delays and congestion but are generally resolved quickly. Traffic is expected to return to normal conditions shortly after the incident has been addressed.",
    ],
    "moderate": [
        "In cases of moderate impact incidents, drivers can expect noticeable disruptions that could affect their estimated arrival time and potentially lead them to consider alternate routes. They might face longer waiting times in traffic queues and have limited access to certain stretches of the road due to ongoing events like accidents or maintenance work.",
        "In moderate impact situations like multiple vehicle collisions or detours due to road maintenance projects, drivers and passengers face noticeable disruptions affecting their estimated arrival time. They could feel annoyed and stressed about finding alternative routes while trying to stay calm as they accommodate unexpected changes in their journey.",
        "Moderate Impact: Incidents classified under this category lead to noticeable disruptions in traffic and can result in moderate delays for drivers. Such incidents might involve multiple vehicle collisions, extended road maintenance projects requiring detours, or weather-related issues like heavy rain causing flooding on certain stretches of the road.",
        "In cases of moderate impact incidents, drivers can expect noticeable disruptions that could affect their estimated arrival time and potentially lead them to consider alternate routes. They might face longer waiting times in traffic queues and have limited access to certain stretches of the road due to ongoing events like accidents or maintenance work.",
        "Moderate incidents typically involve more significant events like multi-vehicle accidents, weather-related issues, or larger roadwork projects that have a noticeable impact on traffic flow. These situations may lead to moderate delays and longer travel times for drivers in the affected area until they are resolved.",
    ],
    "major": [
        "Drivers encountering major impact incidents should be prepared for substantial delays in their journey and possibly rerouting if necessary. Such disruptions may involve detours around closed roads, extended waiting times in heavy traffic, as well as potential stress and frustration caused by the severe disruption in travel plans.",
        "When dealing with major impact incidents like severe accidents or natural disasters blocking major routes, drivers and passengers confront significant delays, detours, and possible cancellations of plans. This scenario can provoke anxiety, irritation, anger, or even panic as they struggle with disrupted schedules and uncertainty surrounding reaching their destination on time.",
        "Major impact incidents are those that cause significant delays and disruptions in the overall traffic flow, often necessitating rerouting of vehicles and substantially increasing travel time for commuters. Causes can range from severe accidents involving multiple vehicles or large-scale construction projects closing off sections of a roadway to natural disasters such as landslides blocking major routes.",
        "Drivers encountering major impact incidents should be prepared for substantial delays in their journey and possibly rerouting if necessary. Such disruptions may involve detours around closed roads, extended waiting times in heavy traffic, as well as potential stress and frustration caused by the severe disruption in travel plans.",
        "Major incidents refer to severe situations like large-scale accidents involving multiple vehicles and injuries, natural disasters, major construction projects, or extreme weather conditions which significantly disrupt traffic flow across a wide area. Such scenarios often result in substantial delays and rerouting of vehicles as emergency services work to address the situation.",
    ],
    "indefinite": [
        "For indefinite impact incidents, drivers will face a high level of uncertainty regarding how much their travel plans will be affected since the duration and extent of such events are unclear at the onset. This could result in unexpected lengthy delays, numerous detours and adjustments to planned routes, or even forced cancellations of trips depending on the severity of the situation they encounter on the road.",
        "For indefinite impact situations where incident duration is highly unpredictable—such as ongoing police investigations following a major accident—drivers and passengers grapple with intense emotions ranging from confusion to apprehension. As they navigate extended delays without clear information on when normal traffic will resume, they may experience feelings of helplessness which could affect morale during the rest of their trip.",
        "The indefinite category pertains to situations where an incident's duration and extent cannot be accurately estimated at the moment due to its complexity and unpredictability. It implies a high level of uncertainty regarding when normal traffic conditions will resume. These could involve ongoing police investigations following a major accident or unpredictable events like wildfires causing extensive road closures until they are contained and deemed safe for travel again.",
        "For indefinite impact incidents, drivers will face a high level of uncertainty regarding how much their travel plans will be affected since the duration and extent of such events are unclear at the onset. This could result in unexpected lengthy delays, numerous detours and adjustments to planned routes, or even forced cancellations of trips depending on the severity of the situation they encounter on the road.",
        "Indefinite incidents represent ongoing issues with no clear resolution timeline—such as long-term construction projects—or unpredictable events like protests and strikes that can affect traffic indefinitely until they conclude. In these cases, drivers should expect extended delays and consider alternative routes or methods of transportation while navigating through the affected areas.",
    ],
}


def convert_probability_to_text(
    value: str,
    plural: bool = False,
    article: bool = False,
) -> str:
    """
    Convert a probability value to a human-readable text representation.

    Args:
        value (str): The probability value to convert.
        plural (bool, optional): If True, the description should be in plural. \
            Defaults to False.
        article (bool, optional): If True, add an article to the description. \
            Defaults to False.

    Returns:
        str: A human-readable text representation of the probability value.

    """
    assert not plural or not article, "Cannot use indefinite article and plural form!"
    vowel_article_text = "an " if article else ""
    plural_text = "s" if plural else ""

    text_map = {
        "rare": f"rare chance of {vowel_article_text}incident{plural_text}",
        "certain": f"certain chance of {vowel_article_text}incident{plural_text}",
        "probable": f"probable chance of {vowel_article_text}incident{plural_text}",
        "risk_of": f"risk of {vowel_article_text}incident{plural_text}",
        "improbable": f"improbable chance of {vowel_article_text}incident{plural_text}",
    }

    return text_map[value]


PROBABILITY_DESCRIPTION = {
    "certain": [
        "A certainty level that indicates a traffic incident has definitely occurred or will occur, based on confirmed information and evidence.",
        "This indicates a traffic incident has definitely occurred or will occur, based on confirmed information and evidence.",
        "There is a certain chance of an incident happening when the occurrence is confirmed by multiple sources, and there's strong evidence supporting its validity.",
        "When there is a certain chance of an incident happening, travelers should expect delays and disruptions, and may need to find alternative routes or modes of transportation to reach their destination safely.",
        "When there is a certain chance of an incident happening, the macroscopic implications may involve widespread traffic disruptions, increased travel times for many commuters, and potential strain on emergency response systems and transportation infrastructure.",
        "When there is a certain chance of an incident happening, the wellbeing of travelers, drivers, and passengers may be directly affected due to possible injuries or accidents. It is crucial for everyone involved to follow safety guidelines, stay alert, and prioritize their physical and mental health during such situations.",
    ],
    "probable": [
        "A certainty level suggesting that a traffic incident is likely to have happened or might happen in the future, based on strong evidence or reliable sources, but not yet completely verified.",
        "This suggests that a traffic incident is likely to have happened or might happen in the future, based on strong evidence or reliable sources, but not yet completely verified.",
        "There is a probable chance of an incident happening when the available information suggests it's likely to have occurred or might occur in the future, but not yet completely verified.",
        "When there is a probable chance of an incident happening, travelers should be prepared for potential difficulties in their journey, such as traffic congestion or detours, and stay updated on the latest information before making any decisions.",
        "When there is a probable chance of an incident happening, the broader consequences might include localized congestion, rerouting in affected areas, heightened vigilance from authorities, and possible adjustments to public transportation schedules or services.",
        "When there is a probable chance of an incident happening, the wellbeing of those on the road may be moderately impacted by potential stress from delays or detours. Travelers should remain vigilant, practice safe driving habits, and consider taking breaks if needed to maintain their overall well-being.",
    ],
    "risk_of": [
        " A certainty level indicating there's a possibility of a traffic incident occurring due to certain conditions or factors, but it is not guaranteed nor fully supported by substantial evidence.",
        "This indicates there's a possibility of a traffic incident occurring due to certain conditions or factors, but it is not guaranteed nor fully supported by substantial evidence.",
        "There is a risk of an incident happening when specific conditions or factors suggest a possibility, but it's not guaranteed nor fully supported by substantial evidence.",
        "When there is a risk of an incident happening, travelers should remain cautious and be aware of the possible issues that could arise during their trip but can continue with their plans while monitoring updates.",
        "When there is a risk of an incident happening, the macroscopic implications could involve precautionary measures taken by local authorities or organizations to minimize impact, increased communication about potential hazards among travelers or residents in the area, and moderate influence on daily routines.",
        "When there is a risk of an incident happening, travelers' wellbeing might be slightly affected as they deal with uncertainty about potential issues during their trip. They should take precautionary measures like staying informed about the situation and planning alternative routes while keeping their emotional state in check.",
    ],
    "improbable": [
        "A certainty level reflecting that a traffic incident is unlikely to have occurred or will occur in the future, as there is little to no supporting evidence and it goes against known patterns or trends.",
        "This reflects that a traffic incident is unlikely to have occurred or will occur in the future, as there is little to no supporting evidence and it goes against known patterns or trends.",
        "There is an improbable chance of an incident happening when the situation has little to no supporting evidence and goes against known patterns or trends, making it unlikely to have occurred or will occur in the future.",
        "When there is an improbable chance of an incident happening, travelers can generally proceed with their planned route or schedule without significant concerns since it's unlikely they will face any major disruptions related to the said event.",
        "When there is an improbable chance of an incident happening, the larger-scale ramifications are generally minimal as it's unlikely that significant disruptions will occur; however, contingency plans may still be developed by relevant stakeholders to prepare for unexpected situations.",
        "When there is an improbable chance of an incident happening, the wellbeing of travelers, drivers, and passengers is generally not significantly impacted since major disruptions are unlikely to occur; however, maintaining awareness and following general safety practices remains important for ensuring a smooth journey.",
    ],
}
