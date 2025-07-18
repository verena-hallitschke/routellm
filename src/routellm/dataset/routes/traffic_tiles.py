"""Module for handling traffic tiles from the Azure traffic API."""

import asyncio
import logging
import random
from collections.abc import Callable
from typing import Any

import aiohttp
import geopandas as gpd
import mapbox_vector_tile
import pandas as pd
import pyproj
from shapely import LineString
from shapely.geometry import shape

from routellm.util.config import Config
from routellm.util.geo_conversion import (
    get_matching_zoom_level,
    get_tile,
    get_tile_projection,
)

DISTANCE_THRESHOLD = 5.0  # in meters


def geo2gdf(geojson: dict[str, Any]) -> gpd.GeoDataFrame:
    """
    Convert a GeoJSON-like dictionary to a GeoDataFrame.

    Args:
        geojson (dict[str, Any]): GeoJSON-like dictionary containing features with \
            geometry and properties.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the geometries and properties from \
            the GeoJSON.

    """
    row_list = []

    for key in geojson:
        for f_ind in range(len(geojson[key]["features"])):
            geometry = shape(geojson[key]["features"][f_ind]["geometry"])
            shape_dict = {
                key: value
                for key, value in geojson[key]["features"][f_ind]["properties"].items()
            }
            shape_dict["geometry"] = geometry
            shape_dict["category_name"] = key

            row_list.append(shape_dict)

    df = pd.DataFrame.from_records(row_list)

    if df.empty:
        gdf = gpd.GeoDataFrame([])
    else:
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    return gdf


async def send_traffic_request(
    query: str,
    tile_transformation_func: Callable[[float, float], tuple[float, float]],
    session: aiohttp.ClientSession,
) -> gpd.GeoDataFrame:
    """
    Load a tile from the Azure traffic API as a GeoDataFrame.

    Args:
        query (str): Query string for the Azure traffic API.
        tile_transformation_func (Callable[[float, float], tuple[float, float]]): \
            Transformation function to convert tile coordinates to longitude and \
            latitude.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.

    Raises:
        aiohttp.ClientOSError: In case of a client-side error during the request.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the features from the tile.

    """
    logger = logging.getLogger()
    request_succeeded = False
    while not request_succeeded:
        try:
            response = await session.get(query)
        except aiohttp.ClientOSError as e:
            if e.errno == 104:
                # Retry
                await asyncio.sleep(5 + random.randint(0, 10))
                continue
            raise e

        if response.status == 200:
            body = await response.read()

            json_f = mapbox_vector_tile.decode(
                body,
                default_options={
                    "y_coord_down": True,
                    "transformer": tile_transformation_func,
                },
            )
            request_succeeded = True
        elif response.status == 429:
            # repeat
            logger.debug("Too many requests")
            await asyncio.sleep(5 + random.randint(0, 10))
        else:
            # Error
            res = gpd.GeoDataFrame()

            error_text = await response.json()

            logger.debug("An error occurred" + str(error_text))
            return res

    return json_f


async def get_flow_tile(
    tile_type: str,
    tile_x: float,
    tile_y: float,
    zoom: int,
    session: aiohttp.ClientSession,
) -> gpd.GeoDataFrame:
    """
    Get a tile describing the traffic flow from the Azure traffic API.

    Args:
        tile_type (str): Type of tile to retrieve (e.g., "microsoft.traffic.delay").
        tile_x (float): X coordinate of the tile.
        tile_y (float): Y coordinate of the tile.
        zoom (int): Zoom level of the tile.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the traffic flow data from the tile.

    """
    conf = Config()
    query = f"https://atlas.microsoft.com/map/tile?api-version=2022-08-01&subscription-key={conf.get('azure-subscription-key')}&tilesetId={tile_type}&zoom={zoom}&x={tile_x}&y={tile_y}"

    res = await send_traffic_request(
        query, get_tile_projection(tile_x, tile_y, zoom), session
    )
    return res


async def get_incident_tile(
    tile_x: float, tile_y: float, zoom: int, session: aiohttp.ClientSession
) -> gpd.GeoDataFrame:
    """
    Get a tile describing traffic incidents from the Azure traffic API.

    Args:
        tile_x (float): X coordinate of the tile.
        tile_y (float): Y coordinate of the tile.
        zoom (int): Zoom level of the tile.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the traffic incident data from the \
            tile.

    """
    conf = Config()
    query = f"https://atlas.microsoft.com/traffic/incident/tile/pbf?api-version=1.0&subscription-key={conf.get('azure-subscription-key')}&zoom={zoom}&x={tile_x}&y={tile_y}"

    res = await send_traffic_request(
        query, get_tile_projection(tile_x, tile_y, zoom), session
    )
    return res


def get_edges_in_osmnx(
    line: LineString, route_df: gpd.GeoDataFrame
) -> tuple[pd.Index, pd.Series]:
    """
    Get the edges corresponding to a line in a GeoDataFrame.

    Args:
        line (LineString): LineString to match with edges in the GeoDataFrame.
        route_df (gpd.GeoDataFrame): GeoDataFrame containing edges with a 'geometry' \
            column.

    Returns:
        tuple[pd.Index, pd.Series]: Tuple containing:
            - pd.Index: Indices of the matched edges.
            - pd.Series: Series of confidence scores for the matched edges.

    """
    # Match segment from azure with osm

    intersections = route_df.intersection(line.buffer(10))
    intersections = intersections[~intersections.apply(lambda x: x.is_empty)]
    intersections = intersections[
        intersections.apply(lambda x: x.geom_type) == "LineString"
    ]

    # calculate distances
    dist = route_df.loc[intersections.index, "geometry"].centroid.distance(line)
    intersections = intersections[dist <= DISTANCE_THRESHOLD]

    return (
        intersections.index,
        (DISTANCE_THRESHOLD - dist.loc[intersections.index]) / DISTANCE_THRESHOLD,
    )


async def get_traffic_flow_info_async(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    crs: pyproj.CRS,
    session: aiohttp.ClientSession,
    tile_type: str = "microsoft.traffic.delay",
) -> gpd.GeoDataFrame:
    """
    Get the traffic flow information for a bounding box from the Azure traffic API.

    Args:
        min_lat (float): Minimum latitude of the bounding box.
        min_lon (float): Minimum longitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.
        max_lon (float): Maximum longitude of the bounding box.
        crs (pyproj.CRS): Coordinate reference system to use for the GeoDataFrame.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.
        tile_type (str, optional): Type of the Azure Maps tile. Defaults to \
            "microsoft.traffic.delay".

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the traffic flow data for the \
            bounding box.

    """
    zoom_level = max(
        0, min(get_matching_zoom_level((min_lat, min_lon), (max_lat, max_lon)) - 1, 22)
    )

    # Get tiles
    min_x, max_y = get_tile((min_lat, min_lon), zoom_level)
    max_x, min_y = get_tile((max_lat, max_lon), zoom_level)

    gdfs = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            flow_tile = await get_flow_tile(tile_type, x, y, zoom_level, session)
            c_gdf = geo2gdf(flow_tile)
            gdfs.append(c_gdf)
    tile_gdf = pd.concat(gdfs)
    tile_gdf = tile_gdf.reset_index(drop=True)
    tile_gdf = tile_gdf.to_crs(crs)

    return tile_gdf


async def get_traffic_incident_info_async(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    crs: pyproj.CRS,
    session: aiohttp.ClientSession,
) -> gpd.GeoDataFrame:
    """
    Get the traffic incident information for a bounding box from the Azure traffic API.

    Args:
        min_lat (float): Minimum latitude of the bounding box.
        min_lon (float): Minimum longitude of the bounding box.
        max_lat (float): Maximum latitude of the bounding box.
        max_lon (float): Maximum longitude of the bounding box.
        crs (pyproj.CRS): Coordinate reference system to use for the GeoDataFrame.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the traffic incident data for the \
            bounding box.

    """
    zoom_level = max(
        0, min(get_matching_zoom_level((min_lat, min_lon), (max_lat, max_lon)) - 1, 22)
    )

    # Get tiles
    min_x, max_y = get_tile((min_lat, min_lon), zoom_level)
    max_x, min_y = get_tile((max_lat, max_lon), zoom_level)

    gdfs = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            incident_tile = await get_incident_tile(x, y, zoom_level, session)
            c_gdf = geo2gdf(incident_tile)
            gdfs.append(c_gdf)
    tile_gdf = pd.concat(gdfs)
    tile_gdf = tile_gdf.reset_index(drop=True)
    tile_gdf = tile_gdf.to_crs(crs)

    return tile_gdf
