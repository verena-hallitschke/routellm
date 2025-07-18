"""Module containing functions to generate task sets."""

import glob
import json
import os
import random
import time
from datetime import datetime
from typing import Any

from tqdm import tqdm

from routellm.dataset.verbalization.tasks import TaskRegistry
from routellm.dataset.verbalization.text_template import splice_route
from routellm.util.geo_conversion import load_gdf
from routellm.util.lm_interfaces.gpt4_interface import GPTInterface

MAX_TOKEN_LENGTH = 2900


def generate_task_set(
    task_name: str,
    route_list: list[str],
    out_path: str,
    use_llm: bool = False,
    show_tqdm: bool = True,
    **kwargs,
) -> str:
    """
    Generate a verbalization for a given task name and list of routes.

    Args:
        task_name (str): Name of the task to generate.
        route_list (list[str]): List of file paths to the routes.
        out_path (str): Output path where the generated tasks will be saved.
        use_llm (bool, optional): If True, use an LLM to generate the responses. \
            Defaults to False.
        show_tqdm (bool, optional): If True, show tqdm progress bar. Defaults to \
            True.
        kwargs: Additional keyword arguments for the GPTInterface.

    Returns:
        str: Path to the directory where the generated tasks are saved.

    """
    gpt_instance = GPTInterface(**kwargs)

    full_outpath = os.path.join(out_path, task_name)
    if not os.path.exists(full_outpath):
        os.makedirs(full_outpath)

    registry = TaskRegistry()

    task = registry[task_name](gpt_instance, use_llm_for_answering=use_llm)
    version = task.VERSION

    for route in tqdm(route_list, disable=not show_tqdm):
        time.sleep(0.2)
        with open(
            os.path.join(os.path.dirname(route), "header.json"), "rt"
        ) as json_file:
            crs = json.load(json_file)["crs"]
        route_df = load_gdf(route, crs, index_cols=["u", "v", "key"])

        task_response_list = task.generate_task(
            route_df,
            route_path=route,
        )

        for t_ind, task_response in enumerate(task_response_list):
            if task_response is None:
                continue

            experiment_name = os.path.basename(os.path.dirname(os.path.dirname(route)))
            city_name = os.path.basename(os.path.dirname(route))

            file_name = (
                f"{task_name}_{t_ind}_{experiment_name}_{city_name}_"
                + f"{os.path.basename(route)[:-4]}.json"
            )
            with open(os.path.join(full_outpath, file_name), "wt") as json_file:
                json.dump(
                    {
                        "file": route,
                        "experiment": experiment_name,
                        "city": city_name,
                        "task_list": [
                            {
                                "task": {
                                    "name": task_name,
                                    "version": version,
                                    "contents": task_response["task"],
                                },
                                "response": task_response["response"],
                            }
                        ],
                        "date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                        "splice_ind": t_ind,
                        "total_splices": len(task_response_list),
                        "route_steps": route_df["step"].to_list(),
                    },
                    json_file,
                    ensure_ascii=False,
                    indent=4,
                )

    return full_outpath


def generate_multi_step_task_set(
    route_list: list[str],
    out_path: str,
    show_tqdm: bool = True,
    min_tasks: int = 1,
    max_tasks: int = 4,
    num_runs: int = 5,
    length_limit: int = 500,
    min_length: int = 50,
    **kwargs,
) -> str:
    """
    Generate a multi-turn conversation for the given routes.

    Args:
        route_list (list[str]): Routes to generate verbalizations for.
        out_path (str): Output path where the generated tasks will be saved.
        show_tqdm (bool, optional): If True, show tqdm progress bar. Defaults to True.
        min_tasks (int, optional): Minimum number of turns. Defaults to 1.
        max_tasks (int, optional): Maximum number of turns. Defaults to 4.
        num_runs (int, optional): Number of runs per route. Defaults to 5.
        length_limit (int, optional): Maximum route length. Defaults to 500.
        min_length (int, optional): Minimum route length. Defaults to 50.
        kwargs: Additional keyword arguments for the GPTInterface.

    Raises:
        ValueError: If the minimum length or maximum tasks are set too high.

    Returns:
        str: Path to the directory where the generated tasks are saved.

    """
    gpt_instance = GPTInterface(**kwargs)

    max_possible_length = min(
        gpt_instance.available_models[gpt_instance.default_model] // 35, length_limit
    )

    full_outpath = os.path.join(out_path, "multi_step")
    if not os.path.exists(full_outpath):
        os.makedirs(full_outpath)

    registry = TaskRegistry()

    available_tasks = list(
        filter(
            lambda x: (
                x
                not in [
                    "llm-tasks.long-description",
                    "llm-tasks.short-description",
                    "llm-tasks.sentence-description",
                    "single-token-task"
                ]
            )
            and "small-" not in x
            and registry[x].SUPPORTS_LLM_POSTPROCESSING,
            registry.get_available_tasks(),
        )
    )

    min_response_length = 500

    if MAX_TOKEN_LENGTH - 50 - min_response_length * max_tasks <= 0:
        raise ValueError("min_length or max_tasks too high!")

    for route in tqdm(route_list, disable=not show_tqdm):
        time.sleep(0.2)

        num_tasks_to_generate = min(
            random.randint(min_tasks, max_tasks), len(available_tasks) - 1
        )

        max_splice_length = min(
            MAX_TOKEN_LENGTH - 50 - min_response_length * num_tasks_to_generate,
            max_possible_length,
        )

        min_spice_length = (
            min_length if min_length < max_splice_length else max_splice_length - 1
        )

        max_splice_length = random.randint(min_spice_length, max_splice_length)

        with open(
            os.path.join(os.path.dirname(route), "header.json"), "rt"
        ) as json_file:
            crs = json.load(json_file)["crs"]

        route_df = load_gdf(route, crs, index_cols=["u", "v", "key"])

        splice_list = [list(range(len(route_df)))]
        if len(route_df) > max_splice_length:
            splice_list = splice_route(len(route_df), max_splice_length)

        if len(splice_list) > num_runs:
            splice_list = random.sample(splice_list, num_runs)

        for spl_ind, splice in enumerate(splice_list):
            splice_df = route_df.iloc[splice]

            # Sample list of tasks
            current_tasks = random.sample(available_tasks, num_tasks_to_generate)

            all_responses = []

            history = []
            style = None

            for task_name in current_tasks:
                task = registry[task_name](
                    gpt_instance, use_llm_for_answering=True, allow_splicing=False
                )

                task_response_list = task.generate_task(
                    splice_df,
                    history=history,
                    style=style,
                    route_path=route,
                )

                if task_response_list is None or len(task_response_list) == 0:
                    continue

                current_response = task_response_list[0]

                if current_response is None:
                    continue

                if style is None:
                    style = current_response["task"]["style"]

                history.append(
                    {"role": "user", "content": current_response["task"]["task_text"]}
                )

                history.append(
                    {"role": "assistant", "content": current_response["response"]}
                )

                all_responses.append(current_response)

            if len(all_responses) == 0:
                continue

            first_resolved_task = all_responses[0]["task"]["task_name"]

            experiment_name = os.path.basename(os.path.dirname(os.path.dirname(route)))
            city_name = os.path.basename(os.path.dirname(route))

            file_name = (
                f"multi_{first_resolved_task}_{spl_ind}_{experiment_name}_"
                + f"{city_name}_{os.path.basename(route)[:-4]}.json"
            )
            with open(os.path.join(full_outpath, file_name), "wt") as json_file:
                json.dump(
                    {
                        "file": route,
                        "experiment": experiment_name,
                        "city": city_name,
                        "task_list": [
                            {
                                "task": {
                                    "name": task_response["task"]["task_name"],
                                    "version": registry[
                                        task_response["task"]["task_name"]
                                    ].VERSION,
                                    "contents": task_response["task"],
                                },
                                "response": task_response["response"],
                            }
                            for task_response in all_responses
                        ],
                        "date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                        "splice_ind": spl_ind,
                        "total_splices": len(splice_list),
                        "route_steps": splice_df["step"].to_list(),
                    },
                    json_file,
                    ensure_ascii=False,
                    indent=4,
                )

    return full_outpath

def single_task_worker(input_dict):
    registry = TaskRegistry()

    task = registry["single-token-task"](None, use_llm_for_answering=False)
    version = task.VERSION

    with open(
        os.path.join(os.path.dirname(input_dict["route_path"]), "header.json"), "rt"
    ) as json_file:
        crs = json.load(json_file)["crs"]
    route_df = load_gdf(input_dict["route_path"], crs, index_cols=["u", "v", "key"])

    task_response_list = task.generate_task(
        route_df, route_path=input_dict["route_path"],
    )

    for t_ind, task_response in enumerate(task_response_list):

        if task_response is None:
            continue

        experiment_name = os.path.basename(os.path.dirname(os.path.dirname(input_dict["route_path"])))
        city_name = os.path.basename(os.path.dirname(input_dict["route_path"]))
        route_name = os.path.basename(input_dict["route_path"])[:-4]

        file_name = f"{t_ind}.json"

        with open(os.path.join(input_dict["output_path"], file_name), "wt") as json_file:
            json.dump(
                {
                    "file": input_dict["route_path"],
                    "experiment": experiment_name,
                    "city": city_name,
                    "task_list": [
                    {
                        "task": {
                            "name": "single-token-task",
                            "version": version,
                            "contents": task_response["task"],
                        },
                        "response": task_response["response"],
                    }
                    ],
                    "date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "splice_ind": t_ind,
                    "total_splices": len(task_response_list),
                },
                json_file,
                ensure_ascii=False,
                indent=4
            )


def run_generation(
    task_name: str,
    gdf_path: str,
    output_path: str,
    task_specific_filtering: bool = False,
    num: int = 100,
    use_llm: bool = False,
    show_tqdm: bool = True,
):
    """
    Run the generation of a task set for a given task name.

    Args:
        task_name (str): Name of the task to generate.
        gdf_path (str): Path to the directory containing GeoDataFrames.
        output_path (str): Path to the directory where the generated tasks will be \
            saved.
        task_specific_filtering (bool, optional): If True, filter tasks specific to \
            the task name. Defaults to False.
        num (int, optional): Number of tasks to generate. Defaults to 100.
        use_llm (bool, optional): If True, use an LLM to generate the responses. \
            Defaults to False.
        show_tqdm (bool, optional): If True, show tqdm progress bar. Defaults to True.

    """
    file_list = list(glob.glob(os.path.join(gdf_path, "*.csv")))

    def filter_fun(sub_list):
        def _fun(x):
            city_name = os.path.basename(os.path.dirname(x))
            experiment_name = os.path.basename(os.path.dirname(os.path.dirname(x)))
            base = os.path.basename(x)[:-4]

            compare_name = f"{experiment_name}_{city_name}_{base}"
            for f in sub_list:
                if compare_name in f:
                    return False
            return True

        return _fun

    c_list = list(
        filter(
            filter_fun(
                list(
                    glob.glob(
                        f"{output_path}/**/{task_name}/*.json"
                        if task_specific_filtering
                        else f"{output_path}/**/*.json",
                        recursive=True,
                    )
                )
            ),
            file_list,
        )
    )
    random.shuffle(c_list)
    generate_task_set(
        task_name,
        c_list[: min(num, len(c_list))],
        out_path=output_path,
        use_llm=use_llm,
        show_tqdm=show_tqdm,
    )


def run_multi_generation(
    gdf_path: str,
    output_path: str,
    num: int = 100,
    show_tqdm: bool = True,
    model: str | None = None,
    length_limit: int = 500,
    num_generations_per_file: int = 5,
    min_length: int = 50,
):
    """
    Run the generation of a multi-turn task set.

    Args:
        gdf_path (str): Path to the directory containing GeoDataFrames.
        output_path (str): Path to the directory where the generated tasks will \
            be saved.
        num (int, optional): Number of tasks to generate. Defaults to 100.
        show_tqdm (bool, optional): If True, show tqdm progress bar. Defaults to True.
        model (str | None, optional): Name of the model to use for generation. \
            If None, use the default model. Defaults to None.
        length_limit (int, optional): Maximum length of the route to generate. \
            Defaults to 500.
        num_generations_per_file (int, optional): Number of generations per file. \
            Defaults to 5.
        min_length (int, optional): Minimum length of the route to generate. \
            Defaults to 50.

    """
    file_list = list(glob.glob(os.path.join(gdf_path, "*.csv")))

    def filter_fun(sub_list):
        def _fun(x):
            city_name = os.path.basename(os.path.dirname(x))
            experiment_name = os.path.basename(os.path.dirname(os.path.dirname(x)))
            base = os.path.basename(x)[:-4]

            compare_name = f"{experiment_name}_{city_name}_{base}"
            for f in sub_list:
                if compare_name in f:
                    return False
            return True

        return _fun

    c_list = list(
        filter(
            filter_fun(
                list(
                    glob.glob(
                        f"{os.path.dirname(output_path)}/**/*.json", recursive=True
                    )
                )
            ),
            file_list,
        )
    )

    kwargs = {}
    if model is not None:
        kwargs["default_model"] = model
    random.shuffle(c_list)
    generate_multi_step_task_set(
        c_list[: min(num, len(c_list))],
        out_path=output_path,
        show_tqdm=show_tqdm,
        length_limit=length_limit,
        num_runs=num_generations_per_file,
        min_length=min_length,
        **kwargs,
    )


def verbalization_worker(input_dict: dict[str, Any]) -> None:
    """
    Worker function to generate verbalizations for a given input dictionary.

    Args:
        input_dict (dict[str, Any]): Input dictionary containing parameters for \
            the generation.

    """
    return run_generation(**input_dict)


def verbalization_worker_multi(input_dict: dict[str, Any]) -> None:
    """
    Worker function to generate multi-turn conversations for a given input dictionary.

    Args:
        input_dict (dict[str, Any]): Input dictionary containing parameters for the \
            generation.

    """
    return run_multi_generation(**input_dict)
