"""Module containing an interface to the osmnx road network."""

from __future__ import annotations

import datetime
import glob
import json
import os
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import tqdm
from networkx import MultiDiGraph, NetworkXNoPath
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from routellm.util.config import Config


class Network:
    """Serializable interface to the road network."""

    DELTA_COEFF = 0.1
    SNAPSHOT_FOLDER_NAME = "graph_snapshots"
    BASE_CLASS_NAME = "base_class.json"
    NODES_FILE_NAME = "nodes.csv"
    EDGES_FILE_NAME = "edges.csv"

    def __init__(
        self,
        center: tuple[float, float],
        bbox_size_sn: float = 0.3,
        bbox_size_we: float = 0.3,
        create_empty: bool = False,
    ):
        """
        Initialize the road network at the given center with the specified box size.

        Args:
            center (tuple[float, float]): Center of the bounding box as (lat, lon).
            bbox_size_sn (float, optional): Size of the bounding box (south, north) in \
                degrees. Defaults to 0.3.
            bbox_size_we (float, optional): Size of the bounding box (east, west) in \
                degrees. Defaults to 0.3.
            create_empty (bool, optional): If True, do not load the road network. \
                Defaults to False.

        """
        if create_empty:
            self.center = None
            self.bbox_size_sn = None
            self.bbox_size_we = None
            self.bbox_poly = None
            self.nodes = None
            self.edges = None
            self.graph = None
            self.graph_path = None
        else:
            self.center = center
            self.bbox_size_sn = bbox_size_sn
            self.bbox_size_we = bbox_size_we

            # Create polygon
            lat, lon = self.center
            delta_sn = self.bbox_size_sn * Network.DELTA_COEFF
            delta_we = self.bbox_size_we * Network.DELTA_COEFF

            self.bbox_poly = Polygon(
                [
                    (
                        lon - 0.5 * bbox_size_we - delta_we,
                        lat + 0.5 * bbox_size_sn + delta_sn,
                    ),
                    (
                        lon - 0.5 * bbox_size_we - delta_we,
                        lat - 0.5 * bbox_size_sn - delta_sn,
                    ),
                    (
                        lon + 0.5 * bbox_size_we + delta_we,
                        lat - 0.5 * bbox_size_sn - delta_sn,
                    ),
                    (
                        lon + 0.5 * bbox_size_we + delta_we,
                        lat + 0.5 * bbox_size_sn + delta_sn,
                    ),
                ]
            )

            self.nodes, self.edges, self.graph = self.create_graph(self.bbox_poly)
            self.graph_path = None

    def create_graph(
        self, bbox_poly: Polygon
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, MultiDiGraph]:
        """
        Load the road network from OpenStreetMap within the given bounding box polygon.

        Args:
            bbox_poly (Polygon): Bounding box polygon to load the road network from.

        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, MultiDiGraph]: Tuple containing:
                - Nodes as a GeoDataFrame.
                - Edges as a GeoDataFrame.
                - NetworkX MultiDiGraph representing the road network.

        """
        conf = Config()
        elevation_path = os.path.join(conf.asset_path, "jaxa_aw3d30", "*.tif")

        net_graph: MultiDiGraph = ox.graph_from_polygon(
            bbox_poly,
            network_type="drive",
            simplify=False,
            retain_all=True,
            truncate_by_edge=True,
        )
        print(net_graph)
        tif_files = glob.glob(elevation_path)

        net_graph = ox.elevation.add_node_elevations_raster(
            net_graph, tif_files, cpus=1
        )
        net_graph = ox.elevation.add_edge_grades(
            net_graph, add_absolute=True
        )  # Adds grade as percentage (height / length)
        net_graph = ox.add_edge_speeds(net_graph, fallback=np.nan)
        net_graph = ox.add_edge_travel_times(net_graph)
        proj_graph = ox.project_graph(net_graph)
        nodes, edges = ox.graph_to_gdfs(proj_graph)

        nodes = nodes.sort_index()
        edges = edges.sort_index()

        landuse = self._get_landuse(edges, bbox_poly)

        edges["landuse"] = landuse.loc[edges.index, "landuse"]
        edges["landuse_distance"] = landuse.loc[edges.index, "distance"]

        return nodes, edges, net_graph

    def _get_landuse(self, edges: gpd.GeoDataFrame, bbox_poly: Polygon) -> pd.DataFrame:
        """
        Load the land use features from OpenStreetMap within the bounding box polygon.

        The land use features are loaded along the edges of the road network.

        Args:
            edges (gpd.GeoDataFrame): Edges of the road network.
            bbox_poly (Polygon): Polygon defining the bounding box for landuse features.

        Returns:
            pd.DataFrame: Land use features with distance to the edge.

        """
        land_feats: gpd.GeoDataFrame = ox.features_from_polygon(
            bbox_poly, {"landuse": True}
        )

        land_feats = land_feats.to_crs(edges.crs)

        land_use = pd.DataFrame(index=edges.index)
        land_use["landuse"] = None
        land_use["distance"] = np.inf

        centroids = edges["geometry"].centroid

        grouped_landuse = land_feats.groupby("landuse").apply(
            lambda x: unary_union(x["geometry"])
        )

        for landuse, geom in tqdm.tqdm(
            grouped_landuse.items(), total=len(grouped_landuse)
        ):
            distances = centroids.distance(geom)

            identifier = distances < land_use["distance"]
            land_use.loc[identifier, "landuse"] = landuse
            land_use.loc[identifier, "distance"] = distances.loc[identifier]

        return land_use

    def get_random_locations(self, num_locations: int) -> np.ndarray:
        """
        Sample random locations within the bounding box of the network.

        Args:
            num_locations (int): Number of random locations to sample.

        Returns:
            np.ndarray: Array of randomly sampled node IDs within the bounding box.

        """
        lat, lon = self.center
        nodes_in_bbox = self.nodes[
            (self.nodes["lon"] < (lon + 0.5 * self.bbox_size_we))
            & (self.nodes["lon"] > (lon - 0.5 * self.bbox_size_we))
            & (self.nodes["lat"] < (lat + 0.5 * self.bbox_size_sn))
            & (self.nodes["lat"] > (lat - 0.5 * self.bbox_size_sn))
        ]
        sample_arr = np.random.choice(
            nodes_in_bbox.index.to_numpy(), num_locations, replace=False
        )

        return sample_arr

    def save(self) -> str:
        """
        Save the road network to a snapshot folder.

        Returns:
            str: Path to the snapshot folder where the network is saved.

        """
        conf = Config()
        output_folder = os.path.join(
            conf.asset_path,
            Network.SNAPSHOT_FOLDER_NAME,
            f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}",
        )
        os.makedirs(output_folder)

        # save without graphs
        edges_temp = self.edges
        nodes_temp = self.nodes

        # Save base class as json header
        prop_dict = {
            "center": self.center,
            "bbox_size_sn": self.bbox_size_sn,
            "bbox_size_we": self.bbox_size_we,
            "bbox_poly": self.bbox_poly.wkt,
            "edges_crs": self.edges.crs.to_json(),
        }
        with open(
            os.path.join(output_folder, Network.BASE_CLASS_NAME), "wt"
        ) as base_file:
            json.dump(prop_dict, base_file)

        # Save edges and nodes
        nodes_temp.to_csv(os.path.join(output_folder, Network.NODES_FILE_NAME))
        edges_temp.to_csv(os.path.join(output_folder, Network.EDGES_FILE_NAME))

        return output_folder

    @staticmethod
    def load(folder: str) -> Network:
        """
        Load a road network from a snapshot folder.

        Args:
            folder (str): Path to the snapshot folder containing the network data.

        Returns:
            Network: Loaded Network instance with the road network data.

        """
        conf = Config()
        loading_path = os.path.join(conf.asset_path, folder)
        # load base class
        with open(
            os.path.join(loading_path, Network.BASE_CLASS_NAME), "rt"
        ) as base_file:
            header_info = json.load(base_file)

        base_graph = Network(None, create_empty=True)

        base_graph.center = header_info["center"]
        base_graph.bbox_size_sn = header_info["bbox_size_sn"]
        base_graph.bbox_size_we = header_info["bbox_size_we"]
        base_graph.bbox_poly = wkt.loads(header_info["bbox_poly"])

        # Load edges and nodes

        nodes = pd.read_csv(
            os.path.join(loading_path, Network.NODES_FILE_NAME),
            index_col="osmid",
            low_memory=False,
        )

        crs = pyproj.CRS.from_json(header_info["edges_crs"])
        nodes["geometry"] = nodes["geometry"].apply(wkt.loads)

        base_graph.nodes = gpd.GeoDataFrame(nodes, crs=crs)

        edges = pd.read_csv(
            os.path.join(loading_path, Network.EDGES_FILE_NAME),
            index_col=["u", "v", "key"],
            low_memory=False,
        )
        edges["geometry"] = edges["geometry"].apply(wkt.loads)
        base_graph.edges = gpd.GeoDataFrame(edges, crs=crs)

        base_graph.graph_path = folder
        base_graph.graph = ox.graph_from_gdfs(
            base_graph.nodes.drop(columns="geometry"), base_graph.edges
        )

        return base_graph

    @staticmethod
    def find_and_load_center(
        center: tuple[float, float],
        bbox_size_sn: float,
        bbox_size_we: float,
        auto_create: bool = True,
        save: bool = True,
    ) -> Network | None:
        """
        Find and load a road network based on the center and bounding box size.

        Args:
            center (tuple[float, float]): Center of the bounding box as (lat, lon).
            bbox_size_sn (float): Bounding box size in the south-north direction.
            bbox_size_we (float): Bounding box size in the west-east direction.
            auto_create (bool, optional): If True create the graph if not found. \
                Defaults to True.
            save (bool, optional): If True, save loaded graph to disk after creation. \
                Defaults to True.

        Returns:
            Network | None: The loaded Network instance if found, otherwise None.

        """
        epslion = min(0.001 * bbox_size_sn, 0.001 * bbox_size_we)
        conf = Config()
        entries = reversed(
            sorted(
                glob.glob(
                    os.path.join(conf.asset_path, Network.SNAPSHOT_FOLDER_NAME, "*/")
                )
            )
        )

        for entry in entries:
            # open base and check bbox size and center
            with open(os.path.join(entry, Network.BASE_CLASS_NAME), "rt") as base_file:
                base_graph: dict[str, Any] = json.load(base_file)

            # Compare bbox size
            if (
                -0.0001
                <= base_graph["bbox_size_sn"] - bbox_size_sn
                < 0.1 * bbox_size_sn
                and -0.0001
                <= base_graph["bbox_size_we"] - bbox_size_we
                < 0.1 * bbox_size_we
            ):
                if (
                    abs(center[0] - base_graph["center"][0]) < epslion
                    and abs(center[1] - base_graph["center"][1]) < epslion
                ):
                    return Network.load(entry)

        if auto_create:
            print(f"No saved file found for {center} - {bbox_size_sn}, {bbox_size_we}.")
            new_network = Network(
                center, bbox_size_sn=bbox_size_sn, bbox_size_we=bbox_size_we
            )
            if save:
                path = new_network.save()
                new_network.graph_path = path
            return new_network

        return None

    def get_coordinates(
        self,
        node_ids: list[int],
        x_coordinate_first: bool = False,
        use_xy: bool = False,
    ) -> list[tuple[float, float]]:
        """
        Get the coordinates of the specified nodes in the network.

        Args:
            node_ids (list[int]): Node IDs to retrieve coordinates for.
            x_coordinate_first (bool, optional): If True, the first entry of the \
                returning tuple is the x/lon coordinate. Defaults to False.
            use_xy (bool, optional): If True, return coordinates in x/y space. \
                Defaults to False.

        Returns:
            list[tuple[float, float]]: List of tuples containing the coordinates of \
                the nodes. Either in (lat, lon) or (y, x) format depending on \
                `use_xy`. If `x_coordinate_first` is True, the order is reversed \
                ((lon, lat)/(x, y)).

        """
        cols = ["lat", "lon"] if not use_xy else ["y", "x"]
        cols = cols[::-1] if x_coordinate_first else cols
        return self.nodes.loc[node_ids, cols].values.tolist()

    def get_route(
        self, start_id: int, end_id: int, weight: str = "length"
    ) -> list[int]:
        """
        Find shortest path between two nodes in the network.

        The shortest path is determined based on the specified weight.

        Args:
            start_id (int): Start OSM node ID.
            end_id (int): End OSM node ID.
            weight (str, optional): Weight, that is minimized. Defaults to "length".

        Returns:
            list[int]: List of node IDs representing the shortest path from start \
                to end.

        """
        paths = ox.shortest_path(self.graph, start_id, end_id, cpus=None, weight=weight)
        if paths is None:
            return []

        try:
            result = list(paths)
        except NetworkXNoPath:
            result = []
        return result

    def get_edge_features(self, node_ids: list[int]) -> gpd.GeoDataFrame:
        """
        Get the features of the edges connecting the specified nodes.

        Args:
            node_ids (list[int]): Node IDs defining the edges.

        Returns:
            gpd.GeoDataFrame: Features of the edges connecting the specified nodes.

        """
        edge_ids = [(node_ids[i], node_ids[i + 1], 0) for i in range(len(node_ids) - 1)]

        return self.edges.loc[edge_ids]
