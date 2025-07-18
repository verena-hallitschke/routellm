"""Constants and utility functions for RouteLLM dataset management."""

from collections import defaultdict

import numpy as np

# See https://developer.tomtom.com/traffic-api/documentation/traffic-incidents/vector-incident-tiles#vector-format

TOMTOM_ICON_CATEGORIES = [
    "Unknown",
    "Accident",
    "Fog",
    "Dangerous Conditions",
    "Rain",
    "Ice",
    "Jam",
    "Lane Closed",
    "Road Closed",
    "Road Works",
    "Wind",
    "Flooding",
    "xxxxxxxxxx",  # 12 is not mapped according to tomtom
    "Cluster",
    "Broken Down Vehicle",
]

ICON_CATEGORIES = [
    "Unknown",
    "Accident",
    "Fog",
    "Dangerous Conditions",
    "Rain",
    "Ice",
    "Jam",
    "Lane Closed",
    "Road Closed",
    "Road Works",
    "Wind",
    "Flooding",
    "Broken Down Vehicle",
]

MAGNITUDES = ["Unknown", "Minor", "Moderate", "Major", "Indefinite"]

PROBABILITIES = [
    "certain",
    "probable",
    "risk_of",
    "improbable",
    "rare",
]  # > 90%, 50-90%, 10-50%, 3-10%, <3%

PROB_INTERVALS = {
    "certain": (0.9, 1.0),
    "probable": (0.5, 0.9),
    "risk_of": (0.1, 0.9),
    "improbable": (0.03, 0.1),
    "rare": (0.0, 0.03),
}


DEFAULT_DATASET = {
    "stuttgart": {
        "center": [48.7831, 9.1815],
        "bbox_size_sn": 0.3,
        "bbox_size_we": 0.3,
        "sample-percentage": 0.0005,
    },
    "hamburg": {
        "center": [53.54862544135612, 9.987518883198955],
        "bbox_size_sn": 0.2,
        "bbox_size_we": 0.4,
        "sample-percentage": 0.0005,
    },
    "munich": {
        "center": [48.134830146609275, 11.58200951453241],
        "bbox_size_sn": 0.4,
        "bbox_size_we": 0.6,
        "sample-percentage": 0.0005,
    },
    "berlin": {
        "center": [52.49352814075898, 13.40660287769637],
        "bbox_size_sn": 0.3,
        "bbox_size_we": 0.45,
        "sample-percentage": 0.0005,
    },
    "cologne": {
        "center": [50.93924114228012, 6.951960595826512],
        "bbox_size_sn": 0.4,
        "bbox_size_we": 0.6,
        "sample-percentage": 0.0005,
    },
    "frankfurt": {
        "center": [50.11434033333651, 8.678513682401714],
        "bbox_size_sn": 0.3,
        "bbox_size_we": 0.6,
        "sample-percentage": 0.0005,
    },
    "duesseldorf": {
        "center": [51.22998584160905, 6.773308341003283],
        "bbox_size_sn": 0.4,
        "bbox_size_we": 0.5,
        "sample-percentage": 0.0005,
    },
    "leipzig": {
        "center": [51.34078559156301, 12.370899243316702],
        "bbox_size_sn": 0.15,
        "bbox_size_we": 0.25,
        "sample-percentage": 0.0005,
    },
    "essen_dortmund": {
        "center": [51.499687031815995, 7.212344504834675],
        "bbox_size_sn": 0.3,
        "bbox_size_we": 0.85,
        "sample-percentage": 0.0005,
    },
    "black_forest": {
        "center": [48.556004363724476, 8.238368371368107],
        "bbox_size_sn": 0.6,
        "bbox_size_we": 0.7,
        "sample-percentage": 0.0001,
    },
    "strassbourg": {
        "center": [48.579491389073844, 7.7496283149865715],
        "bbox_size_sn": 0.3,
        "bbox_size_we": 0.3,
        "sample-percentage": 0.0005,
    },
    "ueberland": {
        "center": [48.25457180991861, 9.222033957609485],
        "bbox_size_sn": 0.7,
        "bbox_size_we": 1.2,
        "sample-percentage": 0.0005,
    },
    "kiel": {
        "center": [54.317471889366246, 10.135485680237082],
        "bbox_size_sn": 0.7,
        "bbox_size_we": 1.2,
        "sample-percentage": 0.0002,
    },
    "hannover": {
        "center": [52.377028093000945, 9.727908551236519],
        "bbox_size_sn": 0.4,
        "bbox_size_we": 0.6,
        "sample-percentage": 0.0001,
    },
    "erfurt": {
        "center": [50.89676246773155, 11.105983479269284],
        "bbox_size_sn": 0.2,
        "bbox_size_we": 0.4,
        "sample-percentage": 0.0003,
    },
    "paris": {
        "center": [48.85804227052771, 2.3499612042993414],
        "bbox_size_sn": 0.5,
        "bbox_size_we": 0.6,
        "sample-percentage": 0.0001,
    },
    "london": {
        "center": [51.50783924051475, -0.1256553546390544],
        "bbox_size_sn": 0.4,
        "bbox_size_we": 0.8,
        "sample-percentage": 0.0001,
    },
    "reykjavik": {
        "center": [64.1018363406349, -21.762066562286833],
        "bbox_size_sn": 0.5,
        "bbox_size_we": 0.6,
        "sample-percentage": 0.0005,
    },
}

MIN_ROUTE_LENGTH = 1000  # in meter

LANDUSE_MAP = defaultdict(
    lambda: "other",
    {
        "error": "other",
        "commercial": "commercial",
        "construction": "construction",
        "education": "institutional",
        "fairground": "commercial",
        "industrial": "industrial",
        "residential": "residential",
        "retail": "commercial",
        "institutional": "institutional",
        "aquaculture": "agriculture",
        "allotments": "greenery",
        "farmland": "agriculture",
        "farmyard": "agriculture",
        "paddy": "agriculture",
        "animal_keeping": "agriculture",
        "flowerbed": "greenery",
        "forest": "forest",
        "greenhouse_horticulture": "agriculture",
        "meadow": "greenery",
        "orchard": "agriculture",
        "plant_nursery": "agriculture",
        "vineyard": "agriculture",
        "basin": "waterbody",
        "reservoir": "waterbody",
        "salt_pond": "waterbody",
        "brownfield": "construction",
        "cemetery": "cemetery",
        "conservation": "other",
        "depot": "transporation",
        "garages": "transporation",
        "grass": "greenery",
        "greenfield": "greenery",
        "landfill": "other",
        "military": "other",
        "port": "waterbody",
        "quarry": "industrial",
        "railway": "transporation",
        "recreation_ground": "greenery",
        "religious": "institutional",
        "village_green": "greenery",
        "winter_sports": "other",
        "user defined": "other",
        "civic_admin": "other",
        "proposed": "other",
        "traffic_island": "other",
        "healthcare": "institutional",
        "harbour": "waterbody",
    },
)

LANDUSE_SUPERCATS = [
    "residential",
    "commercial",
    "agriculture",
    "institutional",
    "waterbody",
    "greenery",
    "construction",
    "forest",
    "industrial",
    "transporation",
    "cemetery",
    "other",
]

LANDUSE_CATEGORIES = [  # unknown
    "error",
    "commercial",
    "construction",
    "education",
    "fairground",
    "industrial",
    "residential",
    "retail",
    "institutional",
    "aquaculture",
    "allotments",
    "farmland",
    "farmyard",
    "paddy",
    "animal_keeping",
    "flowerbed",
    "forest",
    "greenhouse_horticulture",
    "meadow",
    "orchard",
    "plant_nursery",
    "vineyard",
    "basin",
    "reservoir",
    "salt_pond",
    "brownfield",
    "cemetery",
    "conservation",
    "depot",
    "garages",
    "grass",
    "greenfield",
    "landfill",
    "military",
    "port",
    "quarry",
    "railway",
    "recreation_ground",
    "religious",
    "village_green",
    "winter_sports",
    "user defined",
    "civic_admin",
    "proposed",
    "traffic_island",
    "healthcare",
    "harbour",
]

HIGHWAY = [
    "unclassified",
    "secondary",
    "tertiary",
    "residential",
    "primary",
    "primary_link",
    "secondary_link",
    "trunk",
    "living_street",
    "trunk_link",
    "tertiary_link",
    "motorway_link",
    "motorway",
    "busway",
    "disused",
    "rest_area",
    "emergency_bay",
    "road",
]

JUNCTION = [
    "none",
    "roundabout",
    "circular",
    "jughandle",
    "filter",
    "approach",
    "gyratory",
]

SPEED_SIGN = ["none", "sign", "signal", "walk", "unknown"]

ACCESS = [
    "yes",
    "emergency",
    "no",
    "psv",
    "destination",
    "permissive",
    "agricultural",
    "permit",
    "delivery",
    "designated",
    "forestry",
    "hgv",
]

SERVICE = [
    "unknown",
    "parking_aisle",
    "driveway",
    "alley",
    "drive-through",
    "emergency_access",
]

ENCODER_DICT = {
    "landuse": LANDUSE_SUPERCATS,
    "highway": HIGHWAY,
    "incident_category": ICON_CATEGORIES,
}

CAT_FEATURES = list(ENCODER_DICT.keys())

BOOLEAN_VALUES = [
    "oneway",
    "reversed",
    "tunnel",
    "bridge",
    "flow_road_closure",
    "weekday",
    "max_speed_variable",
    "unlimited_speed",
    "lightning",
    "incident_reported",
    "has_junction",
]

CONTINUOUS_COLS = [
    "lanes",
    "length",
    "grade",
    "speed_kph",
    "travel_time",
    "landuse_distance",
    "free_flow_speed",
    "curvature",
    "bearing",
    "delay",
    "incident_delay",
    "incident_distance",
    "current_speed",
    "current_travel_time",
    "cloudCover",
    "temperature",
    "wind_direction",
    "wind_speed",
    "windGust_speed",
    "precipitation",
    "cos_time",
    "sin_time",
    "incident_certainty",
    "incident_magnitude",
]

ENCODER_LIMITS = {}

_running_index = len(CONTINUOUS_COLS) + len(BOOLEAN_VALUES)
for _key, _enc in ENCODER_DICT.items():
    ENCODER_LIMITS[_key] = _running_index
    _running_index += len(_enc)

COL_ORDER = (
    CONTINUOUS_COLS
    + BOOLEAN_VALUES
    + [f"{col}:{val}" for col in ENCODER_DICT for val in ENCODER_DICT[col]]
)


def get_certainty(name: str) -> float:
    """
    Calculate certainty value based on predefined intervals.

    Args:
        name (str): Name of the certainty category.

    Returns:
        float: Certainty value within the specified interval.

    """
    min_v, max_v = PROB_INTERVALS[name]

    val = float(np.random.uniform(min_v, max_v))
    # clip
    val = min(max_v, max(min_v, val))
    return val


def certainty_to_cat(certainty: float) -> str | None:
    """
    Convert a certainty value to a categorical description.

    Args:
        certainty (float): Certainty value in the range [0.0, 1.0].

    Returns:
        str | None: Categorical description of the certainty level, or None if no \
            category matches.

    """
    for name, (min_v, max_v) in PROB_INTERVALS.items():
        if min_v <= certainty < max_v:
            return name

    return None


# Taken from https://en.wikipedia.org/wiki/DBZ_(meteorology)
PRECIPITATION_LIMITS = {
    "no rain": 5.0,
    "trace accumulation or mist": 15.0,
    "trace accumulation": 20.0,
    "light rain": 30.0,
    "light to moderate rain": 35.0,
    "moderate rain": 40.0,
    "moderate to heavy rain": 45.0,
    "heavy rain": 50.0,
    "very heavy rain": None,
}


def convert_precipitation_to_word(value: float) -> str:
    """
    Convert a precipitation value to a categorical description.

    Args:
        value (float): Precipitation value in dBZ (decibels relative to Z).

    Raises:
        ValueError: If the value does not match any category.

    Returns:
        str: Precipitation category as a string.

    """
    # Convert precipitation into categorical value

    for cat, limit in PRECIPITATION_LIMITS.items():
        if limit is None or value < limit:
            return cat

    raise ValueError(f"Invalid precipitation value: {value}. No category found.")


# Taken from https://learningweather.psu.edu/node/38
CLOUD_COVER_LIMITS = {
    "clear": 12.5,
    "mostly clear": 37.5,
    "partly clear": 56.25,
    "partly cloudy": 75.0,
    "mostly cloudy": 95.0,
    "overcast": None,
}


def convert_cloud_cover_to_word(value: float) -> str:
    """
    Convert a cloud cover value to a categorical description.

    Args:
        value (float): Cloud cover value in percentage (0-100).

    Raises:
        ValueError: If the value does not match any category.

    Returns:
        str: Cloud cover category as a string.

    """
    # Convert cloud coverage into categorical value

    for cat, limit in CLOUD_COVER_LIMITS.items():
        if limit is None or value < limit:
            return cat

    raise ValueError(f"Invalid cloud cover value: {value}. No category found.")


# https://www.dwd.de/DE/service/lexikon/Functions/glossar.html?lv3=100390&lv2=100310
# https://en.wikipedia.org/wiki/Beaufort_scale
WIND_FORCE_LIMITS = {
    "calm wind": 1,
    "light air": 5,
    "light breeze": 11,
    "gentle breeze": 19,
    "moderate breeze": 28,
    "fresh breeze": 38,
    "strong breeze": 49,
    "high wind": 61,
    "gale": 74,
    "strong gale": 88,
    "storm": 102,
    "violent storm": 117,
    "hurricane": None,
}


def convert_wind_to_word(value: float) -> str:
    """
    Convert a wind speed value to a categorical description.

    Args:
        value (float): Wind speed value in km/h.

    Raises:
        ValueError: If the value does not match any category.

    Returns:
        str: Wind category as a string.

    """
    # Convert cloud coverage into categorical value

    for cat, limit in WIND_FORCE_LIMITS.items():
        if limit is None or value < limit:
            return cat

    raise ValueError(f"Invalid wind value: {value}. No category found.")


# https://en.wikipedia.org/wiki/Wind_gust
WIND_GUST_LIMITS = {
    "no gusts": 18.52,  # 10 knots
    "gusts": 27.78,  # 15 knots
    "strong gusts": 46.3,  # 25 knots
    "violent gusts": None,
}


def convert_wind_gust_diff_to_word(value: float) -> str:
    """
    Convert a wind gust value to a categorical description.

    Args:
        value (float): Wind gust value in km/h.

    Raises:
        ValueError: If the value does not match any category.

    Returns:
        str: Wind gust category as a string.

    """
    for cat, limit in WIND_GUST_LIMITS.items():
        if limit is None or value < limit:
            return cat

    raise ValueError(f"Invalid wind gust value: {value}. No category found.")


HEADING_LIMITS = {
    "north": 22.5,
    "north east": 67.5,
    "east": 112.5,
    "south east": 157.5,
    "south": 202.5,
    "south west": 247.5,
    "west": 292.5,
    "north west": 337.5,
}


def convert_heading_to_direction(value: float) -> str:
    """
    Convert a heading value to a cardinal direction.

    Args:
        value (float): Heading value in degrees, where 0 is north, 90 is east, 180 is \
            south, and 270 is west.

    Returns:
        str: Cardinal direction corresponding to the heading value.

    """
    for cat, limit in HEADING_LIMITS.items():
        if limit is None or value < limit:
            return cat
    return "north"


UNIT_MAP = {
    "lanes": "lanes",
    "length": "m",
    "grade": "%",
    "speed_kph": "kph",
    "travel_time": "s",
    "landuse_distance": "m",
    "free_flow_speed": "kph",
    "curvature": "1/m",
    "bearing": "degrees",
    "incident_delay": "s",
    "incident_distance": "m",
    "current_speed": "kph",
    "current_travel_time": "s",
    "cloudCover": "%",
    "temperature": "degrees Celsius",
    "wind_direction": "degrees",
    "wind_speed": "kph",
    "windGust_speed": "kph",
    "precipitation": "dBZ",
}
