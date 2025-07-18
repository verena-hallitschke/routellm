"""Module to retrieve weather data along a route using Azure's weather API."""

import asyncio
import logging
import math
import random

import aiohttp
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from routellm.util.config import Config

DEFAULT_SEGMENT_SIZE = 250


async def get_weather_along_route_async(
    route_df: GeoDataFrame,
    session: aiohttp.ClientSession,
    segment_size: float = DEFAULT_SEGMENT_SIZE,
    time_feature: str = "travel_time",
) -> pd.DataFrame:
    """
    Get weather data along a route using Azure's weather API.

    The route will only be sampled every `segment_size` meters.

    Args:
        route_df (GeoDataFrame): GeoDataFrame containing the route data with columns \
            for latitude and longitude.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.
        segment_size (float, optional): The distance in meters between sampled points \
            along the route. Defaults to DEFAULT_SEGMENT_SIZE.
        time_feature (str, optional): Feature used to estimate the time of arrival. \
            Defaults to "travel_time".

    Raises:
        aiohttp.ClientOSError: In case of a client-side error during the request.

    Returns:
        pd.DataFrame: DataFrame containing weather data for the route, with columns:
            - cloud cover
            - temperature
            - wind direction
            - wind speed
            - wind gust speed
            - precipitation
            - lightning count
            - sun glare heading
            - sun glare index

    """
    logger = logging.getLogger()
    conf = Config()

    output_df = pd.DataFrame(index=route_df.index)
    output_df["cloudCover"] = np.nan
    output_df["temperature"] = np.nan
    output_df["wind_direction"] = np.nan
    output_df["wind_speed"] = np.nan
    output_df["windGust_speed"] = np.nan
    output_df["precipitation"] = np.nan
    output_df["lightning_count"] = np.nan
    output_df["sun_glare_heading"] = np.nan
    output_df["sun_glare_index"] = np.nan

    # Split into segments of segment_size in meter
    avg_length = route_df["length"].mean()
    num_entries = max(1, math.floor(segment_size / avg_length))

    selected_entries = route_df.iloc[::num_entries]
    lat_route = selected_entries["from_lat"].values.tolist()
    lon_route = selected_entries["from_lon"].values.tolist()
    time_to_reach = (
        (route_df[time_feature].cumsum() / 60).iloc[::num_entries].to_list()
    )  # estimated time of arrival in minutes

    entry_index = selected_entries.index.to_list()

    time_to_reach = [0.0] + time_to_reach
    if route_df.iloc[::num_entries].iloc[[-1]].index != route_df.iloc[[-1]].index:
        # Add last to entry

        lat_route.append(selected_entries["to_lat"].iloc[-1])
        lon_route.append(selected_entries["to_lon"].iloc[-1])
        entry_index += route_df.iloc[[-1]].index.to_list()
    else:
        # Change last entry to to index
        lat_route[-1] = selected_entries["to_lat"].iloc[-1]
        lon_route[-1] = selected_entries["to_lon"].iloc[-1]
        time_to_reach = time_to_reach[:-1]

    split_ind = [len(lat_route)]

    if len(lat_route) > 60:
        # 60 is limit, split
        num_splits = int(len(lat_route) / 55)
        split_ind = [i * 55 for i in range(1, num_splits + 1)]
        num_splits = int(len(lat_route) / 55)
        split_ind = [i * 55 for i in range(1, num_splits + 1)]
        split_ind.append(len(lat_route))

    entries = list(zip(lat_route, lon_route, time_to_reach))

    prev_split = 0
    for split in split_ind:
        route_str = "".join(
            [f"{lat},{lon},{eta}:" for lat, lon, eta in entries[prev_split:split]]
        )
        route_str = route_str[:-1]  # Cut off colon

        response_succeeded = False

        while not response_succeeded:
            try:
                weather_response_raw = await session.get(
                    "https://atlas.microsoft.com/weather/route/json?api-version=1.1"
                    f"&subscription-key={conf.get('azure-subscription-key')}"
                    f"&query={route_str}"
                )
            except aiohttp.ClientOSError as e:
                if e.errno == 104:
                    # Retry
                    await asyncio.sleep(
                        5 + random.randint(0, 10)
                    )
                    continue
                raise e

            if weather_response_raw.status == 200:
                weather_response = await weather_response_raw.json()
                output_df.loc[entry_index[prev_split:split], "cloudCover"] = [
                    point.get("cloudCover", pd.NA)
                    for point in weather_response.get("waypoints", [])
                ]
                output_df.loc[entry_index[prev_split:split], "temperature"] = [
                    point.get("temperature", {}).get("value", pd.NA)
                    for point in weather_response.get("waypoints", [])
                ]
                output_df.loc[entry_index[prev_split:split], "wind_direction"] = [
                    point.get("wind", {}).get("direction", {}).get("degrees", pd.NA)
                    for point in weather_response.get("waypoints", [])
                ]
                output_df.loc[entry_index[prev_split:split], "wind_speed"] = [
                    point.get("wind", {}).get("speed", {}).get("value", pd.NA)
                    for point in weather_response.get("waypoints", [])
                ]
                output_df.loc[entry_index[prev_split:split], "windGust_speed"] = [
                    point.get("windGust", {}).get("speed", {}).get("value", pd.NA)
                    for point in weather_response.get("waypoints", [])
                ]
                output_df.loc[entry_index[prev_split:split], "precipitation"] = [
                    point.get("precipitation", {}).get("dbz", pd.NA)
                    for point in weather_response.get("waypoints", [])
                ]
                output_df.loc[entry_index[prev_split:split], "lightning_count"] = [
                    point.get("lightningCount", pd.NA)
                    for point in weather_response.get("waypoints", [])
                ]
                output_df.loc[entry_index[prev_split:split], "sun_glare_heading"] = [
                    point.get("sunGlare", {}).get("calculatedVehicleHeading", pd.NA)
                    for point in weather_response.get("waypoints", [])
                ]
                output_df.loc[entry_index[prev_split:split], "sun_glare_index"] = [
                    point.get("sunGlare", {}).get("glareIndex", pd.NA)
                    for point in weather_response.get("waypoints", [])
                ]

                prev_split = split
                prev_split = split

                response_succeeded = True
            elif (
                weather_response_raw.status == 429 or weather_response_raw.status == 500
            ):
                # Too many calls
                logger.debug("Too many requests")
                await asyncio.sleep(5 + random.randint(0, 10))
            else:
                # error
                weather_response = await weather_response_raw.json()
                logger.debug(
                    "An error occurred:" + str(weather_response)
                )
                response_succeeded = True

    return output_df


def get_weather_along_route(
    route_df: GeoDataFrame, segment_size: float = DEFAULT_SEGMENT_SIZE
) -> pd.DataFrame:
    """
    Fetch weather data along a given route.

    Args:
        route_df (GeoDataFrame): GeoDataFrame containing the route data with columns \
            for latitude and longitude.
        segment_size (float, optional): The distance in meters between sampled points \
            along the route. Defaults to DEFAULT_SEGMENT_SIZE.

    Returns:
        pd.DataFrame: The input DataFrame updated with additional columns containing \
            weather information for each route segment.

    """

    async def run_func(
        route_df: pd.DataFrame, segment_size: float = DEFAULT_SEGMENT_SIZE
    ):
        async with aiohttp.ClientSession() as session:
            result = await get_weather_along_route_async(
                route_df, session, segment_size=segment_size
            )
        return result

    loop = asyncio.get_event_loop()
    updated_route_df = loop.run_until_complete(
        asyncio.gather(run_func(route_df, segment_size=segment_size))
    )[0]
    loop.close()

    return updated_route_df
