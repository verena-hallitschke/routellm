# RouteLLM
<!--Add teaser figure-->

## Overview

RouteLLM is a toolkit for generating and processing route datasets for machine learning applications. It provides utilities to download, preprocess, and manage route data, with a focus on datasets from JAXA and related sources.

## Getting Started

### 1. Download JAXA Files

To generate the dataset, you first need to download the required JAXA elevation files from [here](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm). Put the TIF files into the `assets/jaxa_aw3d30/` directory.

### 2. Set-up the Configuration File
Run `set_up.sh` and fill in the keys in the resulting `config.json` file. You can copy the dataset set-up from [`dataset/routes/const.py:DEFAULT_DATASET`](src/routellm/dataset/routes/const.py#L62).

### 3. Install requirements
You can install the project using:

```bash
pip install git+https://github.com/verena-hallitschke/routellm.git
```

## Generate the Dataset

### 1. Generate Static Routes
Generate the static routes using:

```bash
routellm create-static --num-cpus <number_of_workers>
```

### 2. Generate Dynamic Routes
Enrich the static routes with dynamic features using:

```bash
routellm extend-static <static_path>
```

You can check the command-line arguments with:
```bash
routellm extend-static --help
```

### 3. Generate Route Verbalizations
There are two different options to generate the route verbalizations.
The first options is to generate single-turn conversations with:

```bash
routellm run-verbalization <directory_with_routes>
```

The other option is to generate multi-turn conversations with:

```bash
routellm run-verbalization-multiturn <directory_with_routes>
```

Please check the commandline arguments
```bash
routellm run-verbalization --help
routellm run-verbalization-multiturn --help
```

## License

See [LICENSE](LICENSE) for details.

<!-- Add citation -->