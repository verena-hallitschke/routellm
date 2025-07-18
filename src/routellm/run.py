"""Module containing CLI commands for RouteLLM dataset management."""

import datetime
import glob
import json
import multiprocessing
import os
import random
import time
import warnings

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from routellm.dataset.runner.runner import create_static_dataset, extend_static_dataset
from routellm.dataset.verbalization.generate_task_set import (
    run_multi_generation,
    single_task_worker,
    verbalization_worker,
)
from routellm.dataset.verbalization.tasks import TaskRegistry
from routellm.util.config import Config


def error_callback(error: Exception):
    """
    Handle errors in multiprocessing (callback function).

    Args:
        error (Exception): The exception that occurred in the child process.

    """
    import logging  # noqa: PLC0415

    logging.getLogger().error(
        f"Thread {multiprocessing.current_process().name} threw an Exception "
        + f"({type(error).__name__}): {error}"
    )


@click.group()
def cli():
    """RouteLLM CLI."""
    pass


@cli.command()
@click.option(
    "--num-cpus",
    "-c",
    "num_cpus",
    default=max(1, os.cpu_count() - 1),
    help="Number of cpus that will be used. Defaults to os.cpu_count() - 1.",
)
@click.option(
    "--num-runs",
    "-n",
    "num_runs",
    default=1,
    help="Number of times the data in the folders is extended. Defaults to 1.",
)
@click.option(
    "--process-length",
    "-l",
    "process_length",
    default=1,
    help="Number of items each process processes. Defaults to 1.",
)
@click.option(
    "--waiting-period",
    "-w",
    "waiting_period",
    default=0,
    help="Number of seconds to wait inbetween runs. Defaults to 0.",
)
@click.option(
    "--route-limit",
    "-r",
    "route_limit",
    default=None,
    help="Maximum number of routes per folder. Defaults to None.",
)
@click.option(
    "--shuffle/--no-shuffle",
    "shuffle",
    is_flag=True,
    default=True,
    help="Whether the routes within a package should be shuffled. Defaults to True.",
)
@click.option(
    "--check-subfolders/--no-check-subfolders",
    "-s",
    "check_subfolders",
    is_flag=True,
    default=True,
    help="Whether the subfolders of the given ones should be checked for datasets. "
    + "Defaults to True.",
)
@click.option(
    "--run-for",
    "run_for",
    default=0.0,
    help="How long the collection should run in hours. Defaults to running once.",
)
@click.option(
    "--start-in",
    "start_in",
    default=0.0,
    help="In how many hours the collection should start. Defaults to 0.",
)
@click.option(
    "--run-every",
    "run_every",
    default=0.0,
    help="How much stime should be each collection (in hours). Defaults to 0.",
)
@click.argument("paths", nargs=-1)
def extend_static(
    paths,
    num_cpus,
    check_subfolders,
    num_runs,
    process_length,
    waiting_period,
    route_limit,
    shuffle,
    run_for,
    start_in,
    run_every,
):
    """Extend the static dataset with dynamic context features."""
    warnings.simplefilter(
        action="ignore", category=pd.errors.PerformanceWarning
    )

    steps = int(max(1, (start_in * 60 * 60)))

    for _ in tqdm(
        range(steps), total=steps, desc=f"Starting in {steps} s", leave=False
    ):
        time.sleep(1)

    goal_time = time.time() + run_for * 60 * 60  # run for run_for h
    is_first_run = True

    scheduled_wait_time = run_every * 60 * 60

    while time.time() < goal_time or is_first_run:
        is_first_run = False
        start_time = time.time()
        for run in tqdm(
            range(num_runs),
            desc=f"{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            leave=True,
        ):
            if run > 0 and waiting_period > 0:
                waiting_period = max(0, 2.0 * random.random() * waiting_period)

                for _ in tqdm(
                    range(int(waiting_period)),
                    desc=f"Waiting for {int(waiting_period)} s",
                    leave=False,
                ):
                    time.sleep(20)

            timestamp = f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}"

            # Get list of folders
            total_folder_list = []

            for p in paths:
                time.sleep(max(1, round(float(np.random.normal(10, 5)))))
                if os.path.exists(os.path.join(p, "header.json")):
                    total_folder_list.append(p)
                elif check_subfolders:
                    for sp in glob.glob(os.path.join(p, "*/")):
                        if os.path.exists(os.path.join(sp, "header.json")):
                            total_folder_list.append(sp)
                else:
                    print(f"Could not find a dataset in {p}")

            process_packages = []
            total_file_num = 0
            for p in total_folder_list:
                with open(os.path.join(p, "header.json"), "rt") as json_file:
                    header = json.load(json_file)

                crs = header.get("crs")
                if crs is None:
                    print("Missing crs info. Skipping {p}")
                    continue

                files = list(glob.glob(os.path.join(p, "*.csv")))
                city_name = (
                    os.path.basename(os.path.dirname(p))
                    if p[-1] == "/"
                    else os.path.basename(p)
                )

                # create output folder
                conf = Config()
                dynamic_folder_path = os.path.join(
                    conf.asset_path,
                    "routes",
                    "dynamic",
                    timestamp,
                    city_name,
                )

                if not os.path.exists(dynamic_folder_path):
                    os.makedirs(dynamic_folder_path)

                if shuffle:
                    random.shuffle(files)

                if route_limit is not None:
                    files = files[: int(route_limit)]

                for ind in range(0, len(files), process_length):
                    end_ind = min(ind + process_length, len(files))
                    process_packages.append(
                        (city_name, crs, dynamic_folder_path, files[ind:end_ind])
                    )
                total_file_num += len(files)

            random.shuffle(process_packages)

            print(f"Processing {total_file_num} route files")
            with multiprocessing.Pool(processes=num_cpus) as p:
                # p.map(call_extend, [(path, timestamp) for path in total_folder_list])

                for name, crs, dynamic_folder_path, file_list in process_packages:
                    p.apply_async(
                        extend_static_dataset,
                        args=[name, crs, file_list],
                        kwds={"dynamic_folder_path": dynamic_folder_path},
                        error_callback=error_callback,
                    )

                p.close()
                p.join()

        time_diff = np.random.normal(
            scheduled_wait_time, scheduled_wait_time * 0.25
        ) - (time.time() - start_time)

        if time_diff > 0:
            # Start every 2 hours
            time.sleep(time_diff)


@cli.command()
@click.option(
    "--num-cpus",
    "-c",
    "num_cpus",
    default=max(1, os.cpu_count() - 1),
    help="Number of cpus that will be used. Defaults to os.cpu_count() - 1.",
)
def create_static(num_cpus):
    """Collect static route features."""
    # warnings.filterwarnings("error")

    warnings.simplefilter(
        action="ignore", category=pd.errors.PerformanceWarning
    )

    conf = Config()
    timestamp = f"{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}"

    with multiprocessing.Pool(processes=num_cpus) as p:
        for name, settings in conf.get("dataset").items():
            p.apply_async(
                create_static_dataset,
                [
                    {name: settings},
                    settings.get("sample-percentage", conf.get("sample-percentage")),
                    timestamp,
                ],
                error_callback=error_callback,
            )

        try:
            p.close()
            p.join()
        except RuntimeError as e:
            print(
                f"RuntimeError: {e}. This is likely due to a timeout. Please try again."
            )


@cli.command()
@click.argument(
    "route_directory",
    type=click.Path(exists=True, file_okay=False),
    # help="Directory containing route files.",
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(file_okay=False),
    default=os.path.join("assets", "verbalization"),
    help="Directory to save verbalization results.",
)
@click.option(
    "--ignore-tasks",
    multiple=True,
    help="List of tasks to ignore during verbalization.",
)
@click.option(
    "--use-llm", is_flag=True, default=False, help="Use LLM for verbalization."
)
@click.option(
    "--num-cpus",
    "-c",
    "num_cpus",
    default=max(1, os.cpu_count() - 1),
    help="Number of cpus that will be used. Defaults to os.cpu_count() - 1.",
)
@click.option(
    "--num-routes-per-task", default=10, help="Number of routes to process per task."
)
def run_verbalization(
    route_directory,
    output_directory,
    ignore_tasks,
    use_llm,
    num_cpus,
    num_routes_per_task,
):
    """Create route verbalizations."""
    # Randomly selects routes from the given directory and verbalizes them.
    # Uses num_routes_per_task to determine how many routes to process for each task.

    registry = TaskRegistry()

    task_list = [
        {
            "task_name": task,
            "gdf_path": route_directory,
            "output_path": output_directory,
            "num": num_routes_per_task,
            "task_specific_filtering": True,
            "use_llm": use_llm,
            "show_tqdm": False,
        }
        for task in registry.registry
        if task not in ignore_tasks and task != "single-token-task"
    ]

    with multiprocessing.Pool(processes=num_cpus) as p:
        _ = list(
            tqdm(
                p.imap_unordered(verbalization_worker, task_list), total=len(task_list)
            )
        )


@cli.command()
@click.argument(
    "route_directory",
    type=click.Path(exists=True, file_okay=False),
    # help="Directory containing route files.",
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(file_okay=False),
    default=os.path.join("assets", "verbalization"),
    help="Directory to save verbalization results.",
)
@click.option(
    "--num-generations-per-route", default=5, help="Number of generations per route."
)
@click.option(
    "--num-routes", default=10, help="Number of routes to process per conversation."
)
@click.option(
    "--model-name", default=None, help="Name of the model to use for verbalization."
)
def run_verbalization_multiturn(
    route_directory,
    output_directory,
    num_generations_per_route,
    num_routes,
    model_name,
):
    """Create multi-turn route verbalizations."""
    run_multi_generation(
        route_directory,
        output_directory,
        num=num_routes,
        show_tqdm=True,
        model=model_name,
        length_limit=500,
        num_generations_per_file=num_generations_per_route,
        min_length=50,
    )


@cli.command()
@click.argument(
    "route_directory",
    type=click.Path(exists=True, file_okay=False),
    # help="Directory containing route files.",
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(file_okay=False),
    default=os.path.join("assets", "verbalization"),
    help="Directory to save verbalization results.",
)
@click.option(
    "--num-cpus",
    "-c",
    "num_cpus",
    default=max(1, os.cpu_count() - 1),
    help="Number of cpus that will be used. Defaults to os.cpu_count() - 1.",
)
def run_causal_single_token(route_directory, output_directory, num_cpus):
    """Generate single token causal verbalizations."""
    # (gdf_path, output_path, num_workers=47):

    file_list = list(glob.glob(os.path.join(route_directory, "*.csv")))

    task_list = []
    # create folders
    for file in tqdm(file_list, desc="Creating folders"):
        experiment = os.path.basename(os.path.dirname(os.path.dirname(file)))
        city = os.path.basename(os.path.dirname(file))
        route = os.path.basename(file)[:-4]

        if not os.path.exists(os.path.join(output_directory, experiment, city, route)):
            os.makedirs(os.path.join(output_directory, experiment, city, route))

        task_list.append(
            {
                "route_path": file,
                "output_path": os.path.join(output_directory, experiment, city, route),
            }
        )

    with multiprocessing.Pool(processes=num_cpus) as p:
        r = list(
            tqdm(p.imap_unordered(single_task_worker, task_list), total=len(task_list))
        )


if __name__ == "__main__":
    # Start CLI
    cli()
