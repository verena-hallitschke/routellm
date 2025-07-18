"""Module for route feature extraction and sampling from a road network."""

import numpy as np
import tqdm
from geopandas import GeoDataFrame
from shapely.geometry import LineString

from routellm.graph.road_network import Network

MIN_HAUSDORFF_DIST = 0.005


def get_route_features(
    net: Network,
    start: int,
    stop: int,
    columns: list[str] | None = None,
    min_distance: float = MIN_HAUSDORFF_DIST,
) -> list[GeoDataFrame]:
    """
    Retrieve route features between two nodes in a network.

    Searches the shortest route between two OSM nodes in the network and \
    retrieves the features for each route. The function will return a list of \
    GeoDataFrames, each containing the features for one route.

    The function will find a route that minimizes the features specified in \
    `columns` (one route per feature).

    Routes that do nor differ significantly from previously found routes \
    (Hausdorff distance < `min_distance`) will not be returned.

    Args:
        net (Network): Serialized road network.
        start (int): Start OSM node ID.
        stop (int): End OSM node ID.
        columns (list[str] | None, optional): Features used for routing. The \
            function will retrieve a route that minimizes them for each feature. \
            If None, will default to ["length", "travel_time"]. Defaults to None.
        min_distance (float, optional): Difference threshold of two routes. \
            Defaults to MIN_HAUSDORFF_DIST.

    Returns:
        list[GeoDataFrame]: List of GeoDataFrames, each containing the features \
            for one route.

    """
    if columns is None:
        columns = ["length", "travel_time"]

    paths = []
    for col in columns:
        paths.append(net.get_route(start, stop, weight=col))

    lines = []
    features = []

    for path in paths:
        # convert IDs to lat lon
        coordinates = net.get_coordinates(path)
        line_string = LineString(net.get_coordinates(path, x_coordinate_first=True))

        c_df = net.get_edge_features(path)
        c_df["alternative"] = len(lines)
        c_df["step"] = np.arange(len(c_df))
        lats, lons = zip(*coordinates)
        c_df["from_lat"] = lats[:-1]
        c_df["from_lon"] = lons[:-1]
        c_df["to_lat"] = lats[1:]
        c_df["to_lon"] = lons[1:]

        # Compare to other paths and only keep if unique
        if any([line_string.hausdorff_distance(line) < min_distance for line in lines]):
            continue

        lines.append(line_string)
        features.append(c_df)

    return features


def sample_routes(
    net: Network, num_sampled_nodes: int = 20, use_tqdm: bool = True
) -> list[GeoDataFrame]:
    """
    Sample routes from the network by randomly selecting start and goal nodes.

    Args:
        net (Network): Street network to sample routes from.
        num_sampled_nodes (int, optional): Number of nodes to sample. These nodes \
            will be partitioned into start and goal nodes. For each combination \
            of start and goal nodes a route will be sampled. Defaults to 20.
        use_tqdm (bool, optional): Whether to use a tqdm progress bar. \
            Defaults to True.

    Returns:
        list[GeoDataFrame]: List of GeoDataFrames, each containing the features \
            for one route.

    """
    nodes = net.get_random_locations(num_sampled_nodes)

    mid = int(0.5 * num_sampled_nodes)
    start_points = nodes[:mid]
    goal_points = nodes[mid:]

    routes = []
    index = 0

    with tqdm.tqdm(
        total=len(start_points) * len(goal_points), disable=not use_tqdm, leave=True
    ) as pbar:
        for start in start_points:
            for goal in goal_points:
                sub_routes = get_route_features(net, start, goal)

                for route_df in sub_routes:
                    route_df["route"] = index

                pbar.update(1)
                routes.extend(sub_routes)
                index += 1

    return routes
