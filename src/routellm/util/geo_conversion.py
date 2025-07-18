"""Module with utilities for geographic conversions."""

import math
import sys
from collections.abc import Callable
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely import LineString, wkt


def get_matching_zoom_level(
    bbox_min: tuple[float, float],
    bbox_max: tuple[float, float],
    input_crs: str | None = None,
) -> int:
    """
    Calculate which zoom level in Azure Maps is needed so the box can be displayed.

    The lowest zoom level in Azure Maps has a side length of 40075017.0 meters.
    More details here: https://learn.microsoft.com/en-us/azure/azure-maps/zoom-levels-and-tile-grid

    Args:
        bbox_min (tuple[float, float]): Minimal Point of the bounding box
        bbox_max (tuple[float, float]): Maximal point of the bounding box
        input_crs (str | None, optional): Coordinate reference system of the bounding \
            box. If None, it is assumed to be WGS84 (EPSG:4326). Defaults to None.

    Returns:
        int: Zoom level

    """
    if input_crs is None:
        transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857"
        )  # Convert WGS84 to Azure Map system
    else:
        transformer = Transformer.from_crs(
            input_crs, "EPSG:3857"
        )  # Convert WGS84 to Azure Map system

    min_map_point = transformer.transform(bbox_min[1], bbox_min[0])
    max_map_point = transformer.transform(bbox_max[1], bbox_max[0])

    # pylint: disable=unsubscriptable-object
    bbox_width = max_map_point[1] - min_map_point[1]

    # pylint: disable=unsubscriptable-object
    bbox_height = max_map_point[0] - min_map_point[0]

    border_factor = 1.1
    log_max_tile_side = 25.256199796589126  # math.log2(40075017.0)
    level = log_max_tile_side - math.log2(max(bbox_width, bbox_height) * border_factor)

    return math.floor(level)


def get_tile(
    point: tuple[float, float], zoom: int, use_float: bool = False
) -> tuple[float, float]:
    """
    Get the tile coordinates for a given point and zoom level.

    Args:
        point (tuple[float, float]): Point in (lat, lon) format.
        zoom (int): Zoom level.
        use_float (bool, optional): If True, return the tile in float. \
            Defaults to False.

    Returns:
        tuple[float, float]: Tile coordinates in (x, y) format.

    """
    lat, lon = point
    sin_latitude = math.sin(lat * math.pi / 180.0)
    x = ((lon + 180.0) / 360.0) * math.pow(2, zoom)
    y = (
        0.5 - math.log((1.0 + sin_latitude) / (1 - sin_latitude)) / (4.0 * math.pi)
    ) * math.pow(2, zoom)

    if use_float:
        return x, y

    return math.floor(x), math.floor(y)


def get_tile_projection(
    x: float, y: float, zoom: int, extend: int = 4096
) -> Callable[[float, float], tuple[float, float]]:
    """
    Get the projection function for a tile.

    The projection function converts pixel coordinates in the tile to longitude and \
        latitude.

    Args:
        x (float): Tile x coordinate.
        y (float): Tile y coordinate.
        zoom (int): Zoom level.
        extend (int, optional): Extension factor. Defaults to 4096.

    Returns:
        Callable[[float, float], tuple[float, float]]: Projection function that takes \
            pixel coordinates (p_x, p_y) and returns (longitude, latitude).

    """
    x0 = extend * x
    y0 = extend * y
    size = extend * math.pow(2, zoom)

    def project(p_x, p_y):
        y2 = 180.0 - (p_y + y0) * 360.0 / size
        lng = (p_x + x0) * 360.0 / size - 180.0
        lat = 360.0 / math.pi * math.atan(math.exp(y2 * math.pi / 180.0)) - 90.0
        return lng, lat

    return project


def load_gdf(
    path: str, crs_json: dict[str, Any], index_cols: list[str] | None = None
) -> gpd.GeoDataFrame:
    """
    Load a GeoDataFrame from a CSV file.

    Args:
        path (str): Path to the CSV file.
        crs_json (dict[str, Any]): CRS in JSON format.
        index_cols (list[str] | None, optional): Index columns of the csv. \
            Defaults to None.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with the loaded data.

    """
    df = pd.read_csv(path, index_col=index_cols)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS.from_json(crs_json))

    return df


def calculate_curvature(gdf: gpd.GeoDataFrame) -> list[float]:
    """
    Calculate the curvature of each line in a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with lines (key 'geometry').

    Returns:
        list[float]: List of curvature values for each line.

    """
    row_coords = np.array(
        [list(gdf.iloc[i]["geometry"].coords) for i in range(len(gdf))]
    )

    radius_threshold = 100000.0  # approximately straight for 1 km
    epsilon = sys.float_info.epsilon * 100

    v1 = row_coords[1:, 0] - row_coords[:-1, 0]
    v2 = row_coords[1:, 1] - row_coords[1:, 0]
    length = np.linalg.norm(
        row_coords[:-1, 0] - row_coords[1:, 1], axis=1
    )  # length x1 to x3

    # Convert to unit length
    unit_v1 = v1 / np.linalg.norm(v1, axis=1)[:, None]
    unit_v2 = v2 / np.linalg.norm(v2, axis=1)[:, None]

    angles = np.arccos(np.clip(np.sum(unit_v1 * unit_v2, axis=1), -1.0, 1.0))
    menger_curvature = 2.0 * np.sin(angles) / length

    menger_curvature = np.where(
        np.abs(menger_curvature - (1.0 / radius_threshold)) <= epsilon,
        0.0,
        menger_curvature,
    )

    # assume straight line at the beginning and end
    average_curvature = (
        [menger_curvature[0] / 2.0]
        + ((menger_curvature[:-1] + menger_curvature[1:]) / 2.0).tolist()
        + [menger_curvature[-1] / 2.0]
    )
    return average_curvature


def calculate_heading(line: LineString) -> float:
    """
    Calculate the heading of a line.

    Args:
        line (LineString): Line for which to calculate the heading.

    Returns:
        float: Heading in degrees, where 0 is north, 90 is east, 180 is south, \
            and 270 is west.

    """
    x1 = line.coords[0][0]
    x2 = line.coords[-1][0]
    y1 = line.coords[0][1]
    y2 = line.coords[-1][1]
    heading = np.degrees(np.arctan2((x2 - x1), (y2 - y1)))

    if heading < 0.0:
        heading += 360.0

    return heading


def convert_seconds_to_timescale(
    sec: float, time_scale: str = "m", plural: bool = False
) -> str:
    """
    Convert seconds to a human-readable timescale.

    Args:
        sec (float): Time in seconds.
        time_scale (str, optional): Time scale. Can be one of: 's' (seconds), 'm' \
            (minutes), 'h' (hours), 'detailed'(h, m & s). \
            Defaults to "m".
        plural (bool, optional): If true, use plural (i.e. seconds instead of \
            second). Defaults to False.

    Returns:
        str: Human-readable time string.

    """
    actual_timescale = time_scale.lower()
    assert actual_timescale in ["m", "s", "h", "detailed"], (
        f"Unknown time scale {time_scale}!"
    )

    if actual_timescale == "s":
        return f"{round(sec)} second{'' if not plural or round(sec) == 1 else 's'}"

    if actual_timescale == "m":
        minute_val = round(sec / 60.0, 1)
        return f"{minute_val} minute{'' if not plural or minute_val == 1 else 's'}"

    if actual_timescale == "h":
        hour_val = round(sec / (60 * 60), 2)
        return f"{hour_val} hour{'' if not plural or hour_val == 1 else 's'}"

    # get detailed
    remainder = sec
    return_text = ""
    if remainder > 60 * 60:
        hours = int(remainder / (60 * 60))
        remainder -= hours * (60 * 60)
        return_text += f"{hours} hour{'' if not plural or hours == 1 else 's'} "

    if remainder > 60:
        minutes = int(remainder / 60)
        remainder -= minutes * 60
        return_text += f"{minutes} minute{'' if not plural or minutes == 1 else 's'} "

    remainder = round(remainder)
    if remainder > 0:
        return_text += (
            f"{remainder} second{'' if not plural or remainder == 1 else 's'} "
        )

    return return_text.strip()


def convert_meters_to_length_scale(
    dist: float, length_scale: str = "m", plural: bool = False
) -> str:
    """
    Convert a distance in meters to a human-readable length scale.

    Args:
        dist (float): Distance in meters.
        length_scale (str, optional): Length scale. Can be one of: 'm' (meters), \
            'km' (kilometers), 'detailed' (kilometers and meters). Defaults to "m".
        plural (bool, optional): If True, use plural (meters instead of meter). \
            Defaults to False.

    Returns:
        str: Human-readable length string.

    """
    actual_length_scale = length_scale.lower()
    assert actual_length_scale in ["m", "km", "detailed"], (
        f"Unknown time scale {length_scale}!"
    )

    if actual_length_scale == "m":
        return f"{round(dist)} meter{'' if not plural else 's'}"

    if actual_length_scale == "km":
        return f"{round(dist / 1000, 2)} kilometer{'' if not plural else 's'}"

    remainder = dist
    return_text = ""

    if remainder > 1000:
        km = int(remainder / 1000)
        remainder -= km * 1000

        return_text += f"{km} kilometer{'' if not plural or km == 1 else 's'}"

    remainder = round(remainder)
    if remainder > 0:
        return_text += (
            f" {remainder} meter{'' if not plural or remainder == 1 else 's'} "
        )

    return return_text.strip()
