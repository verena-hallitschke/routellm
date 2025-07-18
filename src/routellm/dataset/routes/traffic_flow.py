"""Module to retrieve traffic flow data from Azure for routes."""

import asyncio
import logging
import random
from typing import Any

import aiohttp
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import tqdm
from shapely import geometry, ops

from routellm.util.config import Config
from routellm.util.geo_conversion import get_matching_zoom_level

DISTANCE_THRESHOLD = 5.0  # in meters


async def send_speed_request(
    session: aiohttp.ClientSession, lat: float, lon: float, zoom_level: int
) -> dict[str, Any] | None:
    """
    Send a request to the Azure Maps traffic flow API to get speed data.

    Args:
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.
        lat (float): Latitude of the point to query.
        lon (float): Longitude of the point to query.
        zoom_level (int): Zoom level for the Azure Maps API.

    Raises:
        aiohttp.ClientOSError: In case of a client-side error during the request.

    Returns:
        dict[str, Any] | None: JSON response from the Azure Maps API containing \
            traffic flow data, or None if the request fails.

    """
    logger = logging.getLogger()
    conf = Config()
    query = f"https://atlas.microsoft.com/traffic/flow/segment/json?api-version=1.0&subscription-key={conf.get('azure-subscription-key')}&style=absolute&zoom={zoom_level}&query={lat},{lon}"

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
            request_succeeded = True
        elif response.status == 429:
            # Too manyrequests
            logger.debug("Too many requests")
            await asyncio.sleep(5 + random.randint(0, 10))
        else:
            return None

    json_response = await response.json()

    return json_response


def get_edges_in_osmnx(
    response: dict[str, Any], route_df: gpd.GeoDataFrame
) -> tuple[pd.Index, pd.Series]:
    """
    Get the edges in the GeoDataFrame that match the segments in the response.

    Args:
        response (dict[str, Any]): Response from the Azure traffic API containing flow \
            segment data.
        route_df (gpd.GeoDataFrame): Route GeoDataFrame with OSM IDs and geometry.

    Returns:
        tuple[pd.Index, pd.Series]: Tuple containing:
            - pd.Index: Indices of the route_df that match the segments in the response.
            - pd.Series: Series of weights corresponding to the matched indices.

    """
    # Match segment from azure with osm
    route_line = geometry.LineString(
        [
            [p["longitude"], p["latitude"]]
            for p in response["flowSegmentData"]["coordinates"]["coordinate"]
        ]
    )

    # transform to flat projection
    l_flat = ops.transform(
        pyproj.Transformer.from_crs(
            "EPSG:4326", route_df.crs, always_xy=True
        ).transform,
        route_line,
    )

    buffered_line = l_flat.buffer(10)
    intersections = route_df[route_df.intersects(buffered_line)].intersection(
        buffered_line
    )
    intersections = intersections[~intersections.apply(lambda x: x.is_empty)]
    intersections = intersections[
        intersections.apply(lambda x: x.geom_type) == "LineString"
    ]

    # calculate distances
    dist = route_df.loc[intersections.index, "geometry"].centroid.distance(l_flat)
    intersections = intersections[dist <= DISTANCE_THRESHOLD]

    return intersections.index, DISTANCE_THRESHOLD - dist.loc[intersections.index]


async def calculate_free_flow(
    route_df: gpd.GeoDataFrame,
    session: aiohttp.ClientSession,
    use_tqdm: bool = True,
    zoom: int | None = None,
) -> pd.DataFrame:
    """
    Get free flow speed for the edges in the route_df.

    Args:
        route_df (gpd.GeoDataFrame): GeoDataFrame containing the route data with \
            columns for latitude and longitude.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.
        use_tqdm (bool, optional): Whether to use a tqdm progress bar. Defaults to True.
        zoom (int | None, optional): Zoom level for the Azure Maps API.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the free flow speed for each edge in \
            the route_df. Columns:
                - speed_sum: Sum of the free flow speeds for the edge.
                - normalizer: Normalizer for the speed sum.
                - num_entries: Number of entries for the edge.

    """
    free_flow_speed = pd.DataFrame(index=route_df.index)
    free_flow_speed["speed_sum"] = 0.0
    free_flow_speed["normalizer"] = 0.0
    free_flow_speed["num_entries"] = 0

    # Get traffic info
    centroids = route_df.centroid.to_crs(epsg=4326)
    processed_edges = []

    for num, ind in tqdm.tqdm(enumerate(centroids.index), disable=not use_tqdm):
        # get point in middle

        if ind in processed_edges:
            continue

        point = centroids.loc[ind]

        # get zoom level
        if zoom is None:
            min_x, min_y, max_x, max_y = route_df.loc[ind, "geometry"].bounds
            zoom = min(
                get_matching_zoom_level(
                    (min_y, min_x), (max_y, max_x), input_crs=route_df.crs
                ),
                22,
            )

        response = await send_speed_request(session, point.y, point.x, zoom)

        if response is None:
            continue

        affected_edges, weights = get_edges_in_osmnx(
            response, route_df
        )
        free_flow_speed.loc[affected_edges, "speed_sum"] = (
            free_flow_speed.loc[affected_edges, "speed_sum"]
            + weights * response["flowSegmentData"]["freeFlowSpeed"]
        )
        free_flow_speed.loc[affected_edges, "normalizer"] = (
            free_flow_speed.loc[affected_edges, "normalizer"] + weights
        )
        free_flow_speed.loc[affected_edges, "num_entries"] = (
            free_flow_speed.loc[affected_edges, "num_entries"] + 1
        )

        processed_edges.extend(affected_edges)

    return free_flow_speed


async def get_free_flow_speed_async(
    route_df: gpd.GeoDataFrame,
    session: aiohttp.ClientSession,
    use_tqdm: bool = True,
    zoom: int | None = None,
) -> pd.Series:
    """
    Get the free flow speed for a route.

    Args:
        route_df (gpd.GeoDataFrame): GeoDataFrame containing the route data with \
            columns for latitude and longitude.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.
        use_tqdm (bool, optional): Whether to use a tqdm progress bar. Defaults to True.
        zoom (int | None, optional): Zoom level for the Azure Maps API.

    Returns:
        pd.Series: Series containing the average free flow speed for each edge in the \
            route_df.

    """
    free_flow_speed = await calculate_free_flow(
        route_df, session, use_tqdm=use_tqdm, zoom=zoom
    )

    free_flow_speed["avg_free_flow_speed"] = np.nan
    free_flow_speed.loc[free_flow_speed["num_entries"] > 0, "avg_free_flow_speed"] = (
        free_flow_speed.loc[free_flow_speed["num_entries"] > 0, "speed_sum"]
        / free_flow_speed.loc[free_flow_speed["num_entries"] > 0, "normalizer"]
    )

    return free_flow_speed["avg_free_flow_speed"]


def get_free_flow_speed(
    route_df: gpd.GeoDataFrame, use_tqdm: bool = True, zoom: int | None = None
) -> pd.Series:
    """
    Get the free flow speed for a route.

    Args:
        route_df (gpd.GeoDataFrame): GeoDataFrame containing the route data with \
            columns for latitude and longitude.
        use_tqdm (bool, optional): Whether to use a tqdm progress bar. Defaults to True.
        zoom (int | None, optional): Zoom level for the Azure Maps API.

    Returns:
        pd.Series: Free flow speed for each edge in the route_df.

    """

    async def run_func(
        route_df: gpd.GeoDataFrame, use_tqdm: bool = True, zoom: int | None = None
    ):
        async with aiohttp.ClientSession() as session:
            result = await get_free_flow_speed_async(
                route_df, session, use_tqdm=use_tqdm, zoom=zoom
            )
        return result

    loop = asyncio.get_event_loop()
    net_edges = loop.run_until_complete(
        asyncio.gather(run_func(route_df, use_tqdm=use_tqdm, zoom=zoom))
    )[0]
    loop.close()

    return net_edges
