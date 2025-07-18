"""Module defining route preprocessing functions."""
import datetime
import glob
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder
from tqdm import tqdm

from routellm.dataset.routes.const import (
    COL_ORDER,
    ENCODER_DICT,
    LANDUSE_MAP,
    TOMTOM_ICON_CATEGORIES,
    get_certainty,
)
from routellm.util.geo_conversion import calculate_curvature


def clean_gdf(
    gdf: gpd.GeoDataFrame,
    grade_threshold: float = 0.35,
    distance_threshold: float = 30.0,
    double_digit_grade: bool = False,
    convert_time_zone: bool = True,
) -> gpd.GeoDataFrame | None:
    """
    Clean and preprocess a GeoDataFrame containing road network data.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing road network data.
        grade_threshold (float, optional): Threshold after which a grade will be \
        interpolated. Defaults to 0.35.
        distance_threshold (float, optional): Distance to which the landuse distance \
        will be clipped. Defaults to 30.0.
        double_digit_grade (bool, optional): If True, the return grade in percent. \
        Defaults to False.
        convert_time_zone (bool, optional): If True, convert the timezone. Defaults \
        to True.

    Returns:
        gpd.GeoDataFrame | None: GeoDataFrame with cleaned and preprocessed data, or \
        None if preprocessing fails.

    """
    gdf["new_tunnel"] = False
    gdf.loc[gdf["tunnel"] == "yes", "new_tunnel"] = True
    gdf["tunnel"] = gdf["new_tunnel"]

    gdf["new_bridge"] = False
    gdf.loc[gdf["bridge"] == "yes", "new_bridge"] = True
    gdf["bridge"] = gdf["new_bridge"]

    gdf["oneway"] = gdf["oneway"].fillna(False)
    gdf.loc[gdf["oneway"], "lanes"] = gdf.loc[gdf["oneway"], "lanes"].fillna(1)
    gdf["lanes"] = gdf["lanes"].fillna(2)
    gdf["reversed"] = gdf["reversed"].fillna(False).astype(bool)

    if "flow_road_closure" in gdf.columns:
        gdf["flow_road_closure"] = (
            gdf["flow_road_closure"].fillna(False).astype(bool)
        )
    else:
        return None

    gdf["free_flow_speed"] = gdf["free_flow_speed"].fillna(gdf["speed_kph"])

    if "incident_category" in gdf.columns:
        gdf["incident_category"] = (
            gdf["incident_category"]
            .fillna(0)
            .apply(lambda x: TOMTOM_ICON_CATEGORIES[int(x)])
        )
        gdf["incident_distance"] = gdf["incident_distance"].fillna(-1)
        gdf["incident_certainty"] = (
            gdf["incident_certainty"].fillna("rare").apply(lambda x: get_certainty(x))
        )
        gdf["incident_magnitude"] = (
            gdf["incident_magnitude"].fillna("0").apply(lambda x: int(x))
        )
        gdf["incident_delay"] = gdf["incident_delay"].fillna(0)
        gdf.loc[
            gdf["incident_number_of_reports"] == -1, "incident_number_of_reports"
        ] = 0
        gdf["incident_number_of_reports"] = gdf["incident_number_of_reports"].fillna(0)
        gdf["incident_reported"] = gdf["incident_number_of_reports"] > 0
        gdf.loc[gdf["incident_distance"] > distance_threshold, "incident_distance"] = (
            distance_threshold
        )
    else:
        return None

    gdf["lightning"] = gdf["lightning_count"] > 0

    gdf["junction"] = gdf["junction"].fillna("none")
    gdf["has_junction"] = gdf["junction"] != "none"
    gdf["max_speed_variable"] = False
    gdf["unlimited_speed"] = False

    gdf.loc[gdf["maxspeed"] == "signals", "max_speed_variable"] = True
    gdf.loc[gdf["maxspeed"] == "none", "unlimited_speed"] = True

    gdf.loc[gdf["landuse_distance"] > distance_threshold, "landuse_distance"] = (
        distance_threshold
    )
    gdf["landuse"] = gdf["landuse"].apply(lambda x: LANDUSE_MAP[x])

    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"]) + pd.Series(
        index=gdf.index,
        data=[
            datetime.timedelta(0, x)
            for x in gdf["current_travel_time"].cumsum().shift(fill_value=0.0)
        ],
    )

    # Timestamp was measured on AWS location (Dublin Ireland)
    # -> convert to local time of route
    if convert_time_zone:
        t_finder = TimezoneFinder()
        orig_timezone = t_finder.timezone_at(
            lng=-6.422008381036321, lat=53.28825271941867
        )
        time_zone = t_finder.timezone_at(
            lng=gdf.iloc[0]["from_lon"], lat=gdf.iloc[0]["from_lat"]
        )
        gdf["timestamp"] = (
            gdf["timestamp"].dt.tz_localize(orig_timezone).dt.tz_convert(time_zone)
        )

    gdf["time"] = (
        gdf["timestamp"].dt.hour * (60.0 * 60.0)
        + gdf["timestamp"].dt.minute * 60.0
        + gdf["timestamp"].dt.second
    ) / (24.0 * 60.0 * 60.0)

    # Unique coordinate for each second a day

    gdf["cos_time"] = np.cos(2.0 * np.pi * gdf["time"].to_numpy())
    gdf["sin_time"] = np.sin(2.0 * np.pi * gdf["time"].to_numpy())
    gdf["weekday"] = gdf["timestamp"].dt.day_of_week < 5

    # Convert timestamp

    lengthwise_feature = gdf.copy()
    lengthwise_feature["length_sum"] = (
        lengthwise_feature["length"].cumsum().shift().fillna(0.0)
    )
    lengthwise_feature = lengthwise_feature.set_index("length_sum")
    lengthwise_feature.loc[
        lengthwise_feature["grade_abs"] > grade_threshold, "grade"
    ] = pd.NA

    lengthwise_feature["grade"] = (
        lengthwise_feature["grade"].interpolate(method="index").bfill().ffill()
    )
    lengthwise_feature["cloudCover"] = (
        lengthwise_feature["cloudCover"]
        .interpolate(method="index")
        .bfill()
        .ffill()
        .fillna(0.0)
    )  # / 100.0  # Normalize
    lengthwise_feature["temperature"] = (
        lengthwise_feature["temperature"].interpolate(method="index").bfill().ffill()
    )
    lengthwise_feature["lightning_count"] = (
        lengthwise_feature["lightning_count"]
        .interpolate(method="index")
        .bfill()
        .ffill()
        .fillna(0.0)
    )
    lengthwise_feature["precipitation"] = (
        lengthwise_feature["precipitation"]
        .interpolate(method="index")
        .bfill()
        .ffill()
        .fillna(0.0)
    )  # / 100.0  # Normalize
    lengthwise_feature["wind_speed"] = (
        lengthwise_feature["wind_speed"]
        .interpolate(method="index")
        .bfill()
        .ffill()
        .fillna(0.0)
    )
    lengthwise_feature["wind_direction"] = (
        lengthwise_feature["wind_direction"].interpolate(method="index").bfill().ffill()
    )  # / 360.0  # Normalize
    lengthwise_feature["windGust_speed"] = (
        lengthwise_feature["windGust_speed"]
        .interpolate(method="index")
        .bfill()
        .ffill()
        .fillna(0.0)
    )

    gdf["grade"] = lengthwise_feature["grade"].to_numpy()
    gdf["grade"] = gdf["grade"].bfill().ffill()
    gdf["grade_abs"] = gdf["grade"].abs()

    gdf["cloudCover"] = lengthwise_feature["cloudCover"].to_numpy()
    gdf["temperature"] = lengthwise_feature["temperature"].to_numpy()
    gdf["lightning_count"] = lengthwise_feature["lightning_count"].to_numpy()
    gdf["precipitation"] = lengthwise_feature["precipitation"].to_numpy()
    gdf["wind_speed"] = lengthwise_feature["wind_speed"].to_numpy()
    gdf["wind_direction"] = lengthwise_feature["wind_direction"].to_numpy()
    gdf["windGust_speed"] = lengthwise_feature["windGust_speed"].to_numpy()

    if "curvature" not in gdf.columns:
        gdf["curvature"] = calculate_curvature(gdf)

    gdf.loc[gdf["curvature"] < 1.0 / 100000.0, "curvature"] = 0.0

    x1 = gdf["geometry"].apply(lambda x: x.coords[0][0])
    x2 = gdf["geometry"].apply(lambda x: x.coords[1][0])
    y1 = gdf["geometry"].apply(lambda x: x.coords[0][1])
    y2 = gdf["geometry"].apply(lambda x: x.coords[1][1])

    if "bearing" not in gdf.columns:
        gdf["bearing"] = np.degrees(
            np.arctan2((x2 - x1).to_numpy(), (y2 - y1).to_numpy())
        )
        gdf.loc[gdf["bearing"] < 0.0, "bearing"] = (
            gdf.loc[gdf["bearing"] < 0.0, "bearing"] + 360.0
        )

        # gdf["bearing"] = gdf["bearing"] / 360.0  # Normalize

    if double_digit_grade:
        gdf["grade"] = gdf["grade"] * 100.0
        gdf["grade_abs"] = gdf["grade"].abs()

    return gdf


def preprocess_gdf(
    gdf: gpd.GeoDataFrame,
    grade_threshold: float = 0.35,
    distance_threshold: float = 30.0,
    calculate_counts: bool = True,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame] | None:
    """
    Preprocess a GeoDataFrame containing road network data.

    Cleans the GeoDataFrame, applies one-hot encoding to categorical features,
    and returns the processed GeoDataFrame along with a DataFrame of feature counts.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing road network data.
        grade_threshold (float, optional): Threshold after which a grade will be \
        interpolated. Defaults to 0.35.
        distance_threshold (float, optional): Distance to which the landuse distance \
        will be clipped. Defaults to 30.0.
        calculate_counts (bool, optional): If True, calculate feature counts. Defaults \
        to True.

    Returns:
        tuple[gpd.GeoDataFrame, pd.DataFrame] | None: Tuple containing the processed \
        GeoDataFrame and a DataFrame of feature counts, or None if preprocessing fails.

    """
    gdf = clean_gdf(
        gdf.copy(),
        grade_threshold=grade_threshold,
        distance_threshold=distance_threshold,
    )

    if gdf is None:
        return None

    def get(enc, entry, default_val=0):
        try:
            out = enc.index(entry)

        except ValueError:
            out = default_val
        return out

    # Do one-hot encoding

    count_cols = {}
    new_cols = []
    for col, enc in ENCODER_DICT.items():
        col_names = [f"{col}:{val}" for val in enc]
        if calculate_counts:
            counts = gdf[col].value_counts()
            count_cols[col] = counts
        transformed_col = gdf[col].apply(lambda x: get(enc, x, default_val=0))
        arr = np.eye(len(enc))[transformed_col.to_numpy()]
        df = pd.DataFrame(
            index=gdf.index, data={c: arr[:, i] for i, c in enumerate(col_names)}
        )
        new_cols.append(df)

    full_counts = pd.concat(count_cols) if len(count_cols) > 0 else pd.DataFrame()
    gdf = pd.concat([gdf, *new_cols], axis=1)
    gdf = gdf[COL_ORDER]

    return gdf, full_counts


def load_crs_map(dataset_path: str) -> dict[str, str]:
    """
    Load all CRS definitions from the dataset path and return them as a dictionary.

    Args:
        dataset_path (str): Path to the dataset directory containing subdirectories \
        for each experiment.

    Returns:
        dict[str, str]: Dictionary mapping city names to their CRS definitions.

    """
    crs_list = {}
    exp_list = list(glob.glob(os.path.join(dataset_path, "*/")))

    for experiment_path in tqdm(exp_list, leave=True, position=0):
        experiment_name = os.path.basename(os.path.dirname(experiment_path))

        city_list = list(glob.glob(os.path.join(experiment_path, "*/")))
        for city_path in tqdm(city_list, leave=False, position=1, desc=experiment_name):
            city_name = os.path.basename(os.path.dirname(city_path))

            if crs_list.get(city_name) is None:
                header_path = os.path.join(city_path, "header.json")
                if os.path.exists(header_path):
                    with open(header_path, "rt") as json_file:
                        header = json.load(json_file)
                else:
                    print("No crs")
                    continue  # Missing crs info

                crs_list[city_name] = header["crs"]

    return crs_list
