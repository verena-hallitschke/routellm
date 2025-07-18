"""Module for verbalization of route segments into text descriptions."""

import math
import random

import numpy as np
import pandas as pd
from shapely.ops import linemerge

from routellm.dataset.preprocessing import clean_gdf
from routellm.dataset.routes.const import (
    MAGNITUDES,
    PROB_INTERVALS,
    certainty_to_cat,
    convert_cloud_cover_to_word,
    convert_heading_to_direction,
    convert_precipitation_to_word,
    convert_wind_gust_diff_to_word,
    convert_wind_to_word,
)
from routellm.dataset.verbalization.const_to_text import (
    convert_highway_to_text,
    convert_incident_to_text,
    convert_landuse_to_text,
)
from routellm.util.geo_conversion import calculate_heading, convert_seconds_to_timescale

MIN_OVERLAP_FRACTION = 0.1


def split_route_into_segments(
    route_df: pd.DataFrame,
    gradual_threshold: float = 0.10,
    buffer_segments_to: int = -1,
    max_num_segments: int = -1,
    user_ignore_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Split a route DataFrame into segments based on gradual changes in features.

    Args:
        route_df (pd.DataFrame): DataFrame containing route data with features to \
            analyze.
        gradual_threshold (float, optional): Threshold for gradual columns. A gradual \
            column exceeding this threshold marks the start of a new segment. \
            Defaults to 0.10.
        buffer_segments_to (int, optional): Marks the minimum number of segments \
            if > 0. Defaults to -1.
        max_num_segments (int, optional): Maximum number of segments if > 0. \
            Defaults to -1.
        user_ignore_columns (list[str] | None, optional): Additional columns that \
            will be ignored when determining the new segments. Defaults to None.

    Returns:
        tuple[pd.DataFrame, list[int]]: Tuple containing:
            - DataFrame with segments and aggregated features.
            - List of segment indices corresponding to the original DataFrame.

    """
    if user_ignore_columns is None:
        user_ignore_columns = []

    variable_columns = list(
        filter(
            lambda x: x not in user_ignore_columns,
            [
                "grade_abs",
                "from_lat",
                "from_lon",
                "to_lat",
                "to_lon",
                "bearing",
                "wind_direction",
                "timestamp",
                "osmid",
                "geometry",
                "alternative",
                "step",
                "route",
                "precipitation",
                "cloudCover",
                "wind_speed",
                "windGust_speed",
            ],
        )
    )

    sum_columns = list(
        filter(
            lambda x: x not in user_ignore_columns,
            [
                "length",
                "travel_time",
                "current_travel_time",
            ],
        )
    )  # aggregated with sum

    mean_columns = list(
        filter(
            lambda x: x not in user_ignore_columns,
            [
                "incident_distance",  # mean
                "landuse_distance",  # mean
                "curvature",  # mean
                "grade",
                "precipitation",
                "incident_delay",
                "wind_speed",
                "windGust_speed",
                "cloudCover",
                "bearing",
            ],
        )
    )  # aggregated with mean

    gradual_columns = list(
        filter(
            lambda x: x not in user_ignore_columns,
            [
                # "precipitation",
                # "wind_speed",
                # "windGust_speed",
                "temperature",
                # "cloudCover",
                "delay",
                "current_speed",
                "free_flow_speed",
            ],
        )
    )  # Gradual columns that can change gradually

    ignore_cols = (
        variable_columns
        + sum_columns
        + gradual_columns
        + mean_columns
        + user_ignore_columns
    )

    df = route_df.copy()

    similar_row_index = ~df.drop(columns=ignore_cols, errors="ignore").eq(
        df.drop(columns=ignore_cols, errors="ignore").shift()
    ).all(axis=1)
    combined_segment_index = similar_row_index | (
        (df[gradual_columns] - df[gradual_columns].shift()).abs()
        > gradual_threshold * df[gradual_columns]
    ).any(axis=1)

    df["segment"] = np.nan
    df.loc[combined_segment_index, "segment"] = np.arange(
        len(combined_segment_index[combined_segment_index])
    )
    df["segment"] = df["segment"].ffill().astype("int64")

    if len(combined_segment_index[combined_segment_index]) < buffer_segments_to:
        # Split up segments to reach at least given size
        split_segments = df["segment"].value_counts()
        split_segments = split_segments[split_segments > 1]

        if len(split_segments) > 0:
            num_new_segments = min(
                buffer_segments_to
                - len(combined_segment_index[combined_segment_index]),
                len(split_segments),
            )

            # select random segments to split
            selected_segments = sorted(
                split_segments.sample(num_new_segments, replace=False).index.to_list()
            )
            new_segment_indices = df["segment"].copy()

            for seg in selected_segments:
                possible_vals = df["segment"][df["segment"] == seg].index
                selected_ind = random.randint(1, len(possible_vals) - 1)

                combined_segment_index.loc[possible_vals[selected_ind]] = True
                new_segment_indices.iloc[
                    df.index.get_loc(possible_vals[selected_ind]) :
                ] = (
                    new_segment_indices.iloc[
                        df.index.get_loc(possible_vals[selected_ind]) :
                    ]
                    + 1
                )

            df["segment"] = new_segment_indices

    if (
        max_num_segments > 0
        and len(combined_segment_index[combined_segment_index]) > max_num_segments
    ):
        num_combined_segments = (
            len(combined_segment_index[combined_segment_index]) - max_num_segments
        )

        non_merge_segments = df.loc[similar_row_index[similar_row_index].index][
            "segment"
        ].to_list()
        merge_segments = list(
            filter(
                lambda x: x not in non_merge_segments,
                list(range(1, len(combined_segment_index[combined_segment_index]))),
            )
        )
        selected_segments = sorted(
            np.random.choice(
                merge_segments,
                min(num_combined_segments, len(merge_segments)),
                replace=False,
            ).tolist()
        )
        new_segment_indices = df["segment"].copy()

        for seg in selected_segments:
            first_ind = df.loc[(df["segment"] == seg) & (combined_segment_index)].index[
                0
            ]

            new_segment_indices.iloc[df.index.get_loc(first_ind) :] = (
                new_segment_indices.iloc[df.index.get_loc(first_ind) :] - 1
            )
            combined_segment_index.loc[first_ind] = False

        df["segment"] = new_segment_indices

    segments = df.copy()[combined_segment_index]
    segments["segment"] = df["segment"]

    for col in sum_columns:
        segments[col] = (
            df.groupby("segment", as_index=False)[col]
            .sum()
            .sort_values("segment")[col]
            .to_numpy()
        )

    segments.loc[:, "to_lat"] = segments["from_lat"].shift(
        periods=-1, fill_value=df["to_lat"].iloc[-1]
    )
    segments.loc[:, "to_lon"] = segments["from_lon"].shift(
        periods=-1, fill_value=df["to_lon"].iloc[-1]
    )

    temp_seg = df.copy()
    for col in gradual_columns + mean_columns:
        temp_seg[col] = temp_seg[col] * temp_seg["length"]

        segments[col] = (
            temp_seg.groupby("segment", as_index=False)[col]
            .sum()
            .sort_values("segment")[col]
            .to_numpy()
        )
        segments[col] = segments[col] / segments["length"]

    # Merge geometry
    for ind, g_df in df.groupby("segment", as_index=False):
        merged_line = linemerge(g_df["geometry"].to_list())
        segments.loc[segments["segment"] == ind, "geometry"] = merged_line

    if "bearing" not in user_ignore_columns:
        segments["bearing"] = segments["geometry"].apply(calculate_heading)

    if "grade" not in user_ignore_columns:
        segments["grade_abs"] = segments["grade"].abs()

    segments = segments.reset_index(drop=True)
    segments.index.rename("step", inplace=True)

    if len(user_ignore_columns) > 0:
        segments = segments.drop(columns=user_ignore_columns, errors="ignore")

    return segments, df["segment"].to_list()


def get_row_verbalization(
    step: int,
    row: pd.Series,
    differing_cols: list[str],
    print_street_names: bool = True,
    print_step_id: bool = True,
    original_indices: list[int] | None = None,
    user_ignore_columns: list[str] | None = None,
) -> str:
    """
    Convert a row of route data into a verbal description.

    Args:
        step (int): Number of the step in the route.
        row (pd.Series): Feature values for the step.
        differing_cols (list[str]): Columns that differ from the previous step.
        print_street_names (bool, optional): If True, add street names to the \
            description. Defaults to True.
        print_step_id (bool, optional): If True, print the step ids. Defaults to True.
        original_indices (list[int] | None, optional): List of the indices that are \
            aggregated in this step. Defaults to None.
        user_ignore_columns (list[str] | None, optional): List of columns that will \
            be ignored. Defaults to None.

    Returns:
        str: Verbal description of the step.

    """
    if user_ignore_columns is None:
        user_ignore_columns = []
    # grade_mean = (row["grade"] * row["length"]).sum() / row["length"].sum()
    # curvature_mean = (row["curvature"] * row["length"]).sum() / row["length"].sum()

    step_identifier = f"Step id#{step} has the following measurements:"

    if not print_step_id:
        step_identifier = "This step has the following measurements:"
    elif print_step_id and original_indices is not None:
        if len(original_indices) == 1:
            step_identifier = (
                f"Step {original_indices[0]} has the following measurements:"
            )
        elif len(original_indices) == 2:
            step_identifier = (
                f"Steps {original_indices[0]} and "
                + f"{original_indices[1]} have the following measurements:"
            )
        else:
            step_identifier = (
                f"Steps {original_indices[0]} to "
                + f"{original_indices[-1]} have the following measurements:"
            )

    start = f"{step_identifier} "

    if "length" not in user_ignore_columns:
        start += f"It is {row['length']:0.1f} meters long."

    if "grade" not in user_ignore_columns:
        grade_str = (
            f"{row['grade'] * 100:0.0f} %" if abs(row["grade"]) > 0.01 else "flat"
        )
        start += f" Avg grade: {grade_str}."

    if "curvature" not in user_ignore_columns:
        curvature_str = (
            f"{round(row['curvature'], 5)} 1/m"
            if abs(row["curvature"]) > (1.0 / 100000.0)
            else "straight"
        )
        start += f" Avg curvature: {curvature_str}."

    if "bearing" not in user_ignore_columns and "heading_word" in differing_cols:
        start += f" The step heads towards {row['heading_word']}."

    if "landuse" not in user_ignore_columns:
        start += (
            " Close to the street is "
            + f"{convert_landuse_to_text(row['landuse'], article=True)} (avg "
            + f"{round(row['landuse_distance'])} m away)."
        )

    if (
        "free_flow_speed" not in user_ignore_columns
        and "travel_time" not in user_ignore_columns
        and "current_speed" not in user_ignore_columns
        and "current_travel_time" not in user_ignore_columns
    ):
        start += (
            f" Segment free flow speed: {row['free_flow_speed']: 0.2f} kph "
            + f"({row['travel_time']:0.1f} s), current avg traffic speed: "
            + f"{row['current_speed']: 0.2f} kph (currently "
            + f"{row['current_travel_time']:0.1f} s)."
        )

    if "oneway" not in user_ignore_columns and "oneway" in differing_cols:
        if row["oneway"]:
            start += (
                f" The street {'changes to' if step != 0 else 'is'} a oneway street."
            )
        else:
            start += (
                f" The street is not a oneway street{' anymore' if step != 0 else ''}."
            )

    if "lanes" not in user_ignore_columns and "lanes" in differing_cols:
        start += (
            f" The number of lanes {'changes to' if step != 0 else 'is'} "
            + f"{int(row['lanes'])}."
        )

    if (
        "ref" not in user_ignore_columns
        and "name" not in user_ignore_columns
        and print_street_names
        and ("ref" in differing_cols or "name" in differing_cols)
    ):
        # warnings.warn("Bug that prints unknown as name!")
        ref = (
            row["ref"] if row["ref"] != "Unknown" and not pd.isna(row["ref"]) else None
        )
        name = (
            row["name"]
            if row["name"] != "Unknown" and not pd.isna(row["name"])
            else None
        )

        if ref is not None and name is not None:
            ident = f"{name} ({ref})"
        elif name is not None:
            ident = name
        elif ref is not None:
            ident = ref
        else:
            ident = "unknown"

        if step == 0:
            start += f" The street name is {ident}."
        else:
            start += f" The street name changes to {ident}."

    if "highway" not in user_ignore_columns and "highway" in differing_cols:
        start += (
            f" The street is a {convert_highway_to_text(row['highway'], article=True)}."
        )

    if "reversed" not in user_ignore_columns and "reversed" in differing_cols:
        pass

    if (
        "speed_kph" not in user_ignore_columns
        and "unlimited_speed" not in user_ignore_columns
        and ("speed_kph" in differing_cols or "unlimited_speed" in differing_cols)
    ):
        if row["unlimited_speed"]:
            start += " The allowed speed is unlimited."
        else:
            start += f" The speed limit is {row['speed_kph']:0.2f} kph."

    if (
        "max_speed_variable" not in user_ignore_columns
        and "max_speed_variable" in differing_cols
    ):
        if row["max_speed_variable"]:
            start += " The speed limit is variable."
        else:
            start += " The speed limit is fixed."

    if "bridge" not in user_ignore_columns and "bridge" in differing_cols:
        if row["bridge"]:
            start += " We are passing over a bridge."
        else:
            if step != 0:
                start += " We left the bridge."
            else:
                pass

    if "tunnel" not in user_ignore_columns and "tunnel" in differing_cols:
        if row["tunnel"]:
            start += " We are passing through a tunnel."
        else:
            if step != 0:
                start += " We left the tunnel."
            else:
                pass

    if "has_junction" not in user_ignore_columns and "has_junction" in differing_cols:
        if not row["has_junction"]:
            if step != 0:
                start += " We left the junction."
            else:
                pass
        else:
            start += " This step contains a junction."

    # Incident report
    if (
        "incident_delay" not in user_ignore_columns
        and "incident_certainty" not in user_ignore_columns
        and "incident_category" not in user_ignore_columns
        and "flow_road_closure" not in user_ignore_columns
        and "incident_magnitude" not in user_ignore_columns
        and (
            "incident_category" in differing_cols
            or "flow_road_closure" in differing_cols
        )
        and (
            row["incident_category"] not in ["Unknown", "Cluster"]
            or row["flow_road_closure"]
        )
    ):
        min_c, max_c = PROB_INTERVALS[row["incident_certainty"]]
        cert_name = row["incident_certainty"]

        delay_minutes = round(row["incident_delay"] / 60)
        incident_delay_text = (
            f" (creating a {delay_minutes} minute delay)"
            if round(row["incident_delay"]) > 60.0
            else ""
        )

        certainty_text = (
            f"{cert_name} ({min_c:.0%} to {max_c:.0%}) chance"
            if cert_name != "risk_of"
            else f"risk ({min_c:.0%} to {max_c:.0%})"
        )

        if row["incident_category"] not in ["Unknown", "Cluster"]:
            start += (
                f" There is a {certainty_text} of "
                + f"{convert_incident_to_text(row['incident_category'], article=True)} "
                + f"occurring with a {row['incident_magnitude'].lower()} "
                + f"impact{incident_delay_text}."
            )

        if row["flow_road_closure"]:
            start += " The road is blocked."

    elif (
        "incident_delay" not in user_ignore_columns
        and "incident_certainty" not in user_ignore_columns
        and "incident_category" not in user_ignore_columns
        and "flow_road_closure" not in user_ignore_columns
        and (
            "incident_category" in differing_cols
            or "flow_road_closure" in differing_cols
        )
    ):
        if step != 0:
            start += " The incident has been passed."
        else:
            start += " The road is not blocked."

    if (
        "precipitation" not in user_ignore_columns
        and "cloudCover" not in user_ignore_columns
        and ("precipitation_word" in differing_cols or "cloudCover" in differing_cols)
    ):
        start += (
            f" There is {row['precipitation_word']} "
            + f"({row['precipitation']:0.1f} dbZ) and the sky is "
            + f"{row['cloudCover_word']} ({row['temperature']: 0.1f} deg C)."
        )

    if (
        "wind_speed" not in user_ignore_columns
        and "windGust_speed" not in user_ignore_columns
        and ("wind_word" in differing_cols or "windGust_word" in differing_cols)
    ):
        start += f" There is a {row['wind_word']}, with {row['windGust_word']}."

    if "lightning" not in user_ignore_columns and "lightning" in differing_cols:
        if row["lightning"]:
            start += " There is lightning."
        else:
            start += " There is no lightning."

    return f"{start}\n\n"


def splice_route(
    original_length: int,
    max_num_segments: int,
    min_overlap: float = MIN_OVERLAP_FRACTION,
) -> list[list[int]]:
    """
    Splice a route into segments of a maximum length, ensuring a minimum overlap.

    Args:
        original_length (int): Length of the original segment.
        max_num_segments (int): Maximum number of segments to create.
        min_overlap (float, optional): Minimum overlap between segments as a fraction \
            of `max_num_segments`. Defaults to `MIN_OVERLAP_FRACTION`.

    Returns:
        list[list[int]]: List of lists, where each inner list contains indices of the \
            segments.

    """
    min_overlap = round(min_overlap * max_num_segments)
    num_splits = math.ceil(original_length / (max_num_segments - min_overlap))
    segment_splices = []

    start_ind = 0
    for splice in range(num_splits):
        if splice < num_splits - 1:
            segment_splices.append(
                list(
                    range(start_ind, min(start_ind + max_num_segments, original_length))
                )
            )
        else:
            segment_splices.append(
                list(range(original_length - max_num_segments, original_length))
            )

        start_ind += max_num_segments - min_overlap

    return segment_splices


def route2text(
    route_df: pd.DataFrame,
    buffer_segments_to: int = -1,
    max_num_segments: int = -1,
    print_street_names: bool = True,
    print_step_id: bool = True,
    splice_overflowing_routes: bool = False,
    user_ignore_columns: list[str] | None = None,
    preprocess_df: bool = True,
) -> list[tuple[str, list[str], list[int], list[int]]]:
    """
    Convert a route DataFrame into a list of verbalized segments.

    Args:
        route_df (pd.DataFrame): DataFrame containing route data with features to \
            analyze.
        buffer_segments_to (int, optional): Marks the minimum number of segments \
            if > 0. Defaults to -1.
        max_num_segments (int, optional): Maximum number of segments if > 0. \
            Defaults to -1.
        print_street_names (bool, optional): If True, add street names to the \
            description. Defaults to True.
        print_step_id (bool, optional): If True, print the step ids. Defaults to True.
        splice_overflowing_routes (bool, optional): If True, splice overflowing routes \
            into smaller segments. Defaults to False.
        user_ignore_columns (list[str] | None, optional): Additional columns that \
            will be ignored when determining the new segments. Defaults to None.
        preprocess_df (bool, optional): If True, preprocess the gdf. Defaults to True.

    Returns:
        list[tuple[str, list[str], list[int], list[int]]]: List of tuples, where each \
            tuple contains:
            - Header text for the segment.
            - List of segment texts.
            - List of segment indices corresponding to the original DataFrame.
            - List of splice indices for the segment.

    """
    if user_ignore_columns is None:
        user_ignore_columns = []

    if preprocess_df:
        cleaned_df = clean_gdf(route_df.copy())
    else:
        cleaned_df = route_df.copy()

    if cleaned_df is None:
        return None
    cleaned_df = cleaned_df.drop(
        columns=[
            "area",
            "new_tunnel",
            "new_bridge",
            "cos_time",
            "sin_time",
            "time",
            "x_shape",
            "y_shape",
            "sun_glare_index",
            "sun_glare_heading",
            "incident_cluster_id",
            "junction",
            "access",
            "est_width",
            "maxspeed",
            "lightning_count",
            "width",
            "service",
        ],
        errors="ignore",
    )
    cleaned_df["incident_certainty"] = cleaned_df["incident_certainty"].apply(
        lambda x: certainty_to_cat(x)
    )
    cleaned_df["incident_magnitude"] = cleaned_df["incident_magnitude"].apply(
        lambda x: MAGNITUDES[x]
    )
    cleaned_df["name"] = cleaned_df["name"].fillna("Unknown")
    cleaned_df["ref"] = cleaned_df["ref"].fillna("Unknown")

    # Convert precipitation into categorical value
    cleaned_df["precipitation_word"] = cleaned_df["precipitation"].apply(
        convert_precipitation_to_word
    )

    # Convert cloud coverage into categorical value
    cleaned_df["cloudCover_word"] = cleaned_df["cloudCover"].apply(
        convert_cloud_cover_to_word
    )

    cleaned_df["wind_word"] = cleaned_df["wind_speed"].apply(convert_wind_to_word)
    cleaned_df["windGust_word"] = (
        cleaned_df["windGust_speed"] - cleaned_df["wind_speed"]
    ).apply(convert_wind_gust_diff_to_word)

    cleaned_df["heading_word"] = cleaned_df["bearing"].apply(
        convert_heading_to_direction
    )

    segments, segment_map = split_route_into_segments(
        cleaned_df,
        buffer_segments_to=buffer_segments_to,
        max_num_segments=max_num_segments,
        user_ignore_columns=user_ignore_columns,
    )

    segment_splices = [segments.index]

    if splice_overflowing_routes and len(segments) > max_num_segments:
        segment_splices = splice_route(len(segments), max_num_segments)

    return_texts = []

    arr_segments_map = np.array(segment_map)
    for curr_seg_list in segment_splices:
        curr_seg = segments.loc[curr_seg_list]

        start_row = cleaned_df.iloc[
            np.argwhere(arr_segments_map == curr_seg_list[0]).min()
        ]
        timestamp = start_row["timestamp"]

        header_text = ""

        length_text = ""
        time_text = ""
        if "length" not in user_ignore_columns:
            length_text = (
                f" is a total of {curr_seg['length'].sum() / 1000.0:0.2f} km long"
            )

        if (
            "current_travel_time" not in user_ignore_columns
            and "travel_time" not in user_ignore_columns
        ):
            current_timescale_text = convert_seconds_to_timescale(
                curr_seg["current_travel_time"].sum(),
                time_scale="detailed",
                plural=True,
            )
            timescale_text = convert_seconds_to_timescale(
                curr_seg["travel_time"].sum(), time_scale="detailed", plural=True
            )

            time_text = (
                f" takes {current_timescale_text} ({timescale_text} without traffic)"
            )

        conjunction_text = " and" if length_text != "" and time_text != "" else ""

        header_text = (
            f"The journey starts at {timestamp.strftime('%I:%M %p')} on "
            + f"{'a weekday' if start_row['weekday'] else 'the weekend'},{length_text}"
            + f"{conjunction_text}{time_text}.\n"
        )

        segment_texts = []
        for index, row in curr_seg.eq(curr_seg.shift()).iterrows():
            orig_index = np.argwhere(arr_segments_map == index).flatten().tolist()
            # stop_index = segments.loc[index + 1]["step"] if index
            differing_cols = row[row == False].index.to_list()
            segment_texts.append(
                get_row_verbalization(
                    index,
                    curr_seg.loc[index],
                    differing_cols,
                    print_street_names=print_street_names,
                    print_step_id=print_step_id,
                    original_indices=orig_index,
                    user_ignore_columns=user_ignore_columns,
                )
            )

        splice_indices = cleaned_df.iloc[
            np.argwhere(
                (arr_segments_map >= curr_seg_list[0])
                & (arr_segments_map <= curr_seg_list[-1])
            ).flatten()
        ]["step"].to_list()

        return_texts.append((header_text, segment_texts, segment_map, splice_indices))

    return return_texts
