"""
Module for creating dynamic and static routes.

This module provides functionality to create static routes, extend them with dynamic \
information such as traffic and weather data.
It includes asynchronous operations to fetch data from external APIs.
"""

import asyncio
import datetime
import json
import multiprocessing
import os
from typing import Any

import aiohttp
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import tqdm
from shapely import LineString
from shapely.ops import linemerge

from routellm.dataset.routes.const import MIN_ROUTE_LENGTH
from routellm.dataset.routes.routing import get_route_features
from routellm.dataset.routes.traffic_flow import get_free_flow_speed_async
from routellm.dataset.routes.traffic_tiles import (
    get_traffic_flow_info_async,
    get_traffic_incident_info_async,
)
from routellm.dataset.routes.weather import get_weather_along_route_async
from routellm.graph.road_network import Network
from routellm.util.config import Config
from routellm.util.geo_conversion import (
    calculate_curvature,
    calculate_heading,
    get_matching_zoom_level,
    load_gdf,
)
from routellm.util.logging import set_up_logging


def match_line_to_gdf(
    line: LineString,
    intersection_gdf: gpd.GeoDataFrame,
    buffer_dist: float = 10,
    ignore_points: bool = True,
) -> tuple[gpd.GeoDataFrame, pd.Series]:
    """
    Match a LineString to a GeoDataFrame of intersections.

    Args:
        line (LineString): The LineString to match.
        intersection_gdf (gpd.GeoDataFrame): GeoDataFrame containing the intersections.
        buffer_dist (float, optional): Buffer distance around the line to find \
            intersections. Defaults to 10.
        ignore_points (bool, optional): Whether to ignore points in the intersection \
            GeoDataFrame. Defaults to True.

    Returns:
        tuple[gpd.GeoDataFrame, pd.Series]: A tuple containing:
            - gpd.GeoDataFrame: GeoDataFrame of intersections that match the line.
            - pd.Series: Series of distances from the intersections to the line.

    """
    # Match segment from azure with osm
    buffered_line = line.buffer(buffer_dist)
    intersections = intersection_gdf[
        intersection_gdf.intersects(buffered_line)
    ].intersection(buffered_line)

    if ignore_points:
        intersections = intersections[
            intersections.apply(lambda x: x.geom_type) == "LineString"
        ]

    # calculate distances
    dist = intersection_gdf.loc[intersections.index, "geometry"].distance(line)
    return intersections, dist.loc[intersections.index]


async def create_static_routes(
    net: Network,
    num_sampled_nodes: int,
    folder_path: str,
    session: aiohttp.ClientSession,
    name: str | None = None,
):
    """
    Create static routes for a given network and save them to a folder.

    Args:
        net (Network): Street network to sample routes from.
        num_sampled_nodes (int): Number of nodes to sample.
        folder_path (str): Path to the folder where the routes will be saved.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.
        name (str | None, optional): Name of the dataset. Defaults to None.

    """
    logger = set_up_logging(os.path.join(folder_path, "log.txt"), name)

    nodes = net.get_random_locations(num_sampled_nodes)

    mid = int(0.5 * num_sampled_nodes)
    start_points = nodes[:mid]
    goal_points = nodes[mid:]

    route_ind_arr = np.empty((mid * (num_sampled_nodes - mid), 2), dtype=np.int64)
    route_ind_arr[:, 0] = np.repeat(start_points, (num_sampled_nodes - mid))
    route_ind_arr[:, 1] = np.concatenate([goal_points for _ in range(mid)])

    # Save Header
    header_info = {
        "crs": net.edges.crs.to_json(),
        "net_path": net.graph_path,
        "center": net.center,
        "bbox_size_sn": net.bbox_size_sn,
        "bbox_size_we": net.bbox_size_we,
        "name": name,
        "total_number": mid * (num_sampled_nodes - mid),
    }

    with open(os.path.join(folder_path, "header.json"), "wt") as json_file:
        json.dump(header_info, json_file, ensure_ascii=False, indent=4)

    logger.info(f"Sampling {len(route_ind_arr)} routes")

    use_tqdm = True
    print(" ", end="", flush=True)
    with tqdm.tqdm(
        total=len(start_points) * len(goal_points),
        disable=not use_tqdm,
        desc=f"{name} - static",
        leave=True,
        position=int(multiprocessing.current_process().name.split("-")[1]),
    ) as pbar:
        default_zoom = None
        # Shuffle to have usable routes in case of an error
        for route_index in np.random.permutation(len(route_ind_arr)):
            try:
                start, goal = route_ind_arr[route_index]
                sub_routes = get_route_features(net, start, goal)

                for route_df in sub_routes:
                    if route_df["length"].sum() < MIN_ROUTE_LENGTH:
                        # Discard
                        continue
                    route_df["route"] = route_index
                    route_df["free_flow_speed"] = pd.NA

                    if default_zoom is None:
                        min_x, min_y, max_x, max_y = route_df.loc[
                            route_df["length"].idxmax()
                        ]["geometry"].bounds
                        default_zoom = min(
                            get_matching_zoom_level(
                                (min_y, min_x), (max_y, max_x), input_crs=route_df.crs
                            )
                            - 1,
                            22,
                        )

                    free_flow_speeds = await get_free_flow_speed_async(
                        route_df, session, use_tqdm=False, zoom=default_zoom
                    )
                    route_df.loc[route_df.index, "free_flow_speed"] = (
                        free_flow_speeds.loc[route_df.index]
                    )

                    route_df["curvature"] = calculate_curvature(route_df)
                    x1 = route_df["geometry"].apply(lambda x: x.coords[0][0])
                    x2 = route_df["geometry"].apply(lambda x: x.coords[1][0])
                    y1 = route_df["geometry"].apply(lambda x: x.coords[0][1])
                    y2 = route_df["geometry"].apply(lambda x: x.coords[1][1])
                    route_df["bearing"] = route_df["geometry"].apply(calculate_heading)
                    route_df["x_shape"] = (x2 - x1) / route_df["length"]
                    route_df["y_shape"] = (y2 - y1) / route_df["length"]

                    # Save
                    route_df.to_csv(
                        os.path.join(
                            folder_path,
                            f"route_{route_df['route'].iloc[0]}_{route_df['alternative'].iloc[0]}.csv",
                        )
                    )
                pbar.update(1)
            except Exception as e:
                logger.debug(
                    "Error when processing "
                    f'{route_index} "{route_ind_arr[route_index]}". Dropping route '
                    + f"({type(e).__name__}): {e}"
                )
                pbar.update(1)
                continue


async def add_dynamic_info(
    static_file_list: list[str],
    dynamic_folder_path: str,
    crs_json: str,
    session: aiohttp.ClientSession,
    name: str | None = None,
    use_tqdm: bool = False,
):
    """
    Add dynamic information to static routes.

    Args:
        static_file_list (list[str]): List of paths to static route files.
        dynamic_folder_path (str): Path to the folder where dynamic data will be saved.
        crs_json (str): Coordinate reference system in JSON format.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.
        name (str | None, optional): Name of the dataset. Defaults to None.
        use_tqdm (bool, optional): Whether to use a tqdm progress bar. \
            Defaults to False.

    """
    logger = set_up_logging(os.path.join(dynamic_folder_path, "log.txt"), name)
    static_features = None

    distance_threshold = 5  # in meter
    crs = pyproj.CRS.from_json(crs_json)
    # Add traffic separately due to processing time
    print(" ", end="", flush=True)
    with tqdm.tqdm(
        total=len(static_file_list),
        disable=not use_tqdm,
        desc=f"{name} - dynamic",
        leave=True,
        position=max(1, int(multiprocessing.current_process().name.split("-")[1])),
    ) as pbar:
        for index, file_name in enumerate(static_file_list):
            try:
                route_df = load_gdf(file_name, crs_json, index_cols=["u", "v", "key"])

                if static_features is None and not os.path.exists(
                    os.path.join(dynamic_folder_path, "header.json")
                ):
                    static_features = route_df.columns
                    header_info = {
                        "crs": crs_json,
                        "city": name,
                        "static_features": static_features.to_list(),
                        "dynamic_features": list(
                            filter(
                                lambda col: col not in static_features, route_df.columns
                            )
                        ),
                    }

                    with open(
                        os.path.join(dynamic_folder_path, "header.json"), "wt"
                    ) as json_file:
                        json.dump(header_info, json_file, indent=4, ensure_ascii=False)

                bounds = route_df["geometry"].to_crs(epsg=4326).bounds
                traffic_flow_tiles = await get_traffic_flow_info_async(
                    bounds["miny"].min(),
                    bounds["minx"].min(),
                    bounds["maxy"].max(),
                    bounds["maxx"].max(),
                    crs,
                    session,
                )
                traffic_incident_tiles_cor = get_traffic_incident_info_async(
                    bounds["miny"].min(),
                    bounds["minx"].min(),
                    bounds["maxy"].max(),
                    bounds["maxx"].max(),
                    crs,
                    session,
                )
                route_df = route_df.sort_index()

                route_line = linemerge(route_df["geometry"].to_list())
                flow_intersections, flow_distances = match_line_to_gdf(
                    route_line, traffic_flow_tiles, buffer_dist=20
                )

                reduced_flow_tiles = traffic_flow_tiles.loc[
                    flow_intersections.index
                ].sort_index()

                route_df["delay"] = 1.0
                route_df["flow_road_closure"] = None

                for ind, row in route_df.iterrows():
                    flow_distances = reduced_flow_tiles.distance(row["geometry"])
                    if flow_distances.min() <= distance_threshold:
                        min_entry = flow_distances.idxmin()
                        if "traffic_level" in reduced_flow_tiles.columns:
                            route_df.loc[ind, "delay"] = reduced_flow_tiles.loc[
                                min_entry
                            ]["traffic_level"]
                        if "road_closure" in reduced_flow_tiles.columns:
                            route_df.loc[ind, "flow_road_closure"] = (
                                reduced_flow_tiles.loc[min_entry]["road_closure"]
                            )

                traffic_incident_tiles = await traffic_incident_tiles_cor
                incident_intersections, flow_distances = match_line_to_gdf(
                    route_line,
                    traffic_incident_tiles,
                    buffer_dist=20,
                    ignore_points=False,
                )

                reduced_incident_tiles = traffic_incident_tiles.loc[
                    incident_intersections.index
                ].sort_index()
                route_df["incident_category"] = None
                route_df["incident_delay"] = np.nan
                route_df["incident_certainty"] = None
                route_df["incident_number_of_reports"] = -1
                route_df["incident_magnitude"] = 0
                route_df["incident_distance"] = np.nan
                route_df["incident_cluster_id"] = None

                icon_category_columns = list(
                    filter(
                        lambda x: "icon_category" in x, reduced_incident_tiles.columns
                    )
                )

                for ind, row in reduced_incident_tiles.iterrows():
                    affected_edges, distances = match_line_to_gdf(
                        row["geometry"], route_df, ignore_points=False
                    )
                    affected_edges = affected_edges[distances < distance_threshold]
                    affected_edges = affected_edges[
                        route_df.loc[affected_edges.index, "incident_distance"].isnull()
                        | route_df.loc[affected_edges.index, "incident_distance"]
                        > distances.loc[affected_edges.index]
                    ].index
                    if affected_edges.empty:
                        continue
                    categories = list(
                        filter(
                            lambda x: not np.isnan(x),
                            [row[c] for c in icon_category_columns],
                        )
                    )
                    route_df.loc[affected_edges, "incident_category"] = (
                        categories[0] if len(categories) > 0 else None
                    )
                    route_df.loc[affected_edges, "incident_delay"] = (
                        row["delay"]
                        if "delay" in reduced_incident_tiles.columns
                        else np.nan
                    )
                    route_df.loc[affected_edges, "incident_certainty"] = (
                        row["probability_of_occurrence"]
                        if "probability_of_occurrence" in reduced_incident_tiles.columns
                        else None
                    )
                    route_df.loc[affected_edges, "incident_number_of_reports"] = (
                        row["number_of_reports"]
                        if "number_of_reports" in reduced_incident_tiles.columns
                        else -1
                    )
                    route_df.loc[affected_edges, "incident_magnitude"] = (
                        row["magnitude"]
                        if "magnitude" in reduced_incident_tiles.columns
                        else 0
                    )
                    route_df.loc[affected_edges, "incident_distance"] = distances.loc[
                        affected_edges
                    ]

                    if "clustered" in reduced_incident_tiles.columns and not np.isnan(
                        row["clustered"]
                    ):
                        route_df.loc[affected_edges, "incident_cluster_id"] = row[
                            "clustered"
                        ]
                    elif "cluster_id" in reduced_incident_tiles.columns:
                        route_df.loc[affected_edges, "incident_cluster_id"] = row[
                            "cluster_id"
                        ]

                speed_estimate = route_df[
                    "free_flow_speed"
                ].copy()
                outlier_threshold = (
                    (route_df["free_flow_speed"] - route_df["speed_kph"])
                    .abs()
                    .quantile(0.95)
                )
                speed_estimate[
                    (
                        (route_df["free_flow_speed"] - route_df["speed_kph"]).abs()
                        > outlier_threshold
                    )
                    & (route_df["maxspeed"] != "none")
                    & (route_df["maxspeed"] != "signal")
                ] = pd.NA
                speed_estimate = speed_estimate.fillna(route_df["speed_kph"])
                route_df["current_speed"] = speed_estimate * route_df["delay"]
                route_df["current_travel_time"] = (
                    1.0 / (route_df["current_speed"] / 3.6)
                ) * route_df["length"]
                route_df.loc[
                    np.isinf(route_df["current_travel_time"]), "current_travel_time"
                ] = pd.NA
                route_df["current_travel_time"] = route_df[
                    "current_travel_time"
                ].fillna(route_df["travel_time"])
                route_df = route_df.sort_values(by="step")
                time_feature = "current_travel_time"

                weather_df = await get_weather_along_route_async(
                    route_df, session, segment_size=75, time_feature=time_feature
                )
                route_df["timestamp"] = (
                    datetime.datetime.now()
                )

                for col in weather_df.columns:
                    route_df[col] = weather_df.loc[route_df.index, col]

                pbar.update(1)
                # Save
                route_df.to_csv(
                    os.path.join(
                        dynamic_folder_path,
                        f"routes_{route_df['route'].iloc[0]}_{route_df['alternative'].iloc[0]}.csv",
                    )
                )
            except Exception as e:
                logger.debug(
                    f'Error when processing {index} at "{file_name}". '
                    + f"Dropping route ({type(e).__name__}): {e}"
                )
                continue


async def extend_static_dataset_async(
    name: str,
    crs_json: str,
    route_list: list[str],
    session: aiohttp.ClientSession,
    timestamp: str = f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}",
    dynamic_folder_path: str | None = None,
) -> str:
    """
    Extend a static dataset with dynamic information such as traffic and weather data.

    Args:
        name (str): Name of the dataset.
        crs_json (str): Coordinate reference system in JSON format.
        route_list (list[str]): List of paths to static route files.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.
        timestamp (str, optional): Timestamp for the dataset. Defaults to \
            f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}".
        dynamic_folder_path (str | None): Path to the folder where dynamic data will \
            be saved.

    Returns:
        str: Path to the folder containing the extended dataset with dynamic data.

    """
    if dynamic_folder_path is None:
        conf = Config()
        folder_name = os.path.join(
            conf.asset_path,
            "routes",
        )
        dynamic_folder_path = os.path.join(
            folder_name,
            "dynamic",
            timestamp,
            name,
        )

        if not os.path.exists(dynamic_folder_path):
            os.makedirs(dynamic_folder_path)

    # Load gpd files
    await add_dynamic_info(
        route_list,
        dynamic_folder_path,
        crs_json,
        session,
        name=name,
    )

    return dynamic_folder_path


def extend_static_dataset(
    name: str,
    crs_json: str,
    route_list: list[str],
    timestamp: str = f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}",
    dynamic_folder_path: str | None = None,
) -> list[str]:
    """
    Extend a static dataset with dynamic information such as traffic and weather data.

    Args:
        name (str): Name of the dataset.
        crs_json (str): Coordinate reference system in JSON format.
        route_list (list[str]): List of paths to static route files.
        timestamp (str, optional): Timestamp for the dataset. Defaults to \
            f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}".
        dynamic_folder_path (str | None): Path to the folder where dynamic data will \
            be saved.

    Returns:
        list[str]: List of paths to the files containing the extended dataset with \
            dynamic data.

    """

    async def run_func(
        name: str,
        crs_json,
        route_list: list[str],
        timestamp: str,
        dynamic_folder_path: str,
    ):
        async with aiohttp.ClientSession() as session:
            result = await extend_static_dataset_async(
                name,
                crs_json,
                route_list,
                session,
                timestamp=timestamp,
                dynamic_folder_path=dynamic_folder_path,
            )

        return result

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    result_list = loop.run_until_complete(
        asyncio.gather(
            run_func(name, crs_json, route_list, timestamp, dynamic_folder_path)
        )
    )[0]

    loop.close()

    return result_list


async def create_static_dataset_async(
    center: tuple[float, float],
    name: str,
    session: aiohttp.ClientSession,
    bbox_size_sn: float = 0.3,
    bbox_size_we: float = 0.3,
    node_sample_percentage: float = 0.0002,
    timestamp: str = f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}",
) -> str:
    """
    Create a static dataset of routes around a given center point.

    Args:
        center (tuple[float, float]): Center point (latitude, longitude) around which \
            to create routes.
        name (str): Name of the dataset.
        session (aiohttp.ClientSession): Session for making asynchronous HTTP requests.
        bbox_size_sn (float, optional): Size of the bounding box in the north-south \
            direction. Defaults to 0.3.
        bbox_size_we (float, optional): Size of the bounding box in the west-east \
            direction. Defaults to 0.3.
        node_sample_percentage (float, optional): Percentage of nodes to sample from \
            the network. Defaults to 0.0002.
        timestamp (str, optional): Timestamp for the dataset. Defaults to \
            f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}".

    Returns:
        str: Path to the folder containing the static dataset with routes.

    """
    conf = Config()
    folder_name = os.path.join(
        conf.asset_path,
        "routes",
    )

    static_folder_path = os.path.join(
        folder_name,
        "static",
        timestamp,
        f"{name}",
    )

    os.makedirs(static_folder_path)

    # print(f'Creating dataset "{name}"')
    net = Network.find_and_load_center(
        center, bbox_size_sn=bbox_size_sn, bbox_size_we=bbox_size_we
    )

    num_routes = max(
        2, round(len(net.nodes) * node_sample_percentage)
    )

    await create_static_routes(net, num_routes, static_folder_path, session, name=name)

    return static_folder_path


def create_static_dataset(
    dataset: dict[str, Any], node_sample_percentage: float, timestamp: str
) -> list[str]:
    """
    Create a static dataset of routes around given centers.

    Args:
        dataset (dict[str, Any]): Dictionary containing dataset information, where \
            keys are dataset names and values are dictionaries with keys:
            - "center": tuple of (latitude, longitude) for the center point.
            - "bbox_size_sn": size of the bounding box in the north-south direction.
            - "bbox_size_we": size of the bounding box in the west-east direction.
        node_sample_percentage (float): Percentage of nodes to sample from the network.
        timestamp (str): Timestamp for the dataset, used to create unique folder names.

    Returns:
        list[str]: List of paths to the folders containing the static datasets with \
            routes.

    """

    async def run_func(dataset, node_sample_percentage, timestamp):
        async with aiohttp.ClientSession() as session:
            results = []
            for name, dataset_info in dataset.items():
                res = await create_static_dataset_async(
                    dataset_info["center"],
                    name,
                    session,
                    bbox_size_sn=dataset_info["bbox_size_sn"],
                    bbox_size_we=dataset_info["bbox_size_we"],
                    node_sample_percentage=node_sample_percentage,
                    timestamp=timestamp,
                )

                results.append(res)

            return results

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    result_list = loop.run_until_complete(
        asyncio.gather(run_func(dataset, node_sample_percentage, timestamp))
    )[0]

    loop.close()

    return result_list
