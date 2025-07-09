# Lectures on Monte Carlo Theory


|  Book title&nbsp;&nbsp;&nbsp;&nbsp;       |Lectures on Monte Carlo Theory  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;              |
|--------------------|------------------------------------------------|
| **Authors**       | Paweł Lorek and Tomasz Rolski                                 |
| **Affiliation**   | University of Wrocław, Poland                                          |
| **Publisher**     | Springer, 2025                                                 |
| **Springer link** | [link.springer.com/book/9783032011893](https://link.springer.com/book/9783032011893) |
| **Series**        | *Probability Theory and Stochastic Modelling*                 |
| **ISBN**          | 978-3-032-01189-3                                              |




This repository contains Python scripts and code examples for the Monte Carlo simulations presented in the above book.

 
## About the Code

Most examples should run out-of-the-box, provided you are using Python 3.12. The numerical values and figures in the book can be reproduced directly from the scripts. All required packages are listed in `requirements.txt`.


To ensure full compatibility, it is recommended to use the exact package versions provided, preferably by setting up the environment via Conda (see instructions below). If you encounter issues, you can also use the provided Docker image for a fully reproducible setup.


## Prerequisites
- Python 3.12.2
- Conda (recommended), pip, or Docker




## Installation
### 1. Using Conda (Recommended)

If you don’t have Conda installed, we recommend installing **Miniconda**, a lightweight version of Anaconda:

- Download **Miniconda** for your system: https://docs.conda.io/en/latest/miniconda.html
- Follow the installation instructions for your operating system.
  On Linux/macOS, restart your terminal after installation.
  On Windows, open the **Anaconda Prompt** after installation.

We provide two versions of the Conda environment:

- `environment.yml` – minimal version with essential packages
- `environment_full.yml` – full version capturing the entire working environment used during development

To create and activate the environment:

```bash
conda env create -f environment.yml
conda activate MonteCarlo_book
```


> ⚠️  If you experience missing packages, layout issues, unexpected errors, or if the results are not
> correctly reproduced, we recommend using the full environment file for complete reproducibility:


```bash
conda env create -f environment_full.yml
conda activate MonteCarlo_book
```


### 2. Using pip
```bash
pip install -r requirements.txt
```
 
## Repository Structure
```
Lectures_on_Monte_Carlo_Theory/
├── chapter_1/
│   ├── ch1_2d_game_winning_prob.py
│   ├── ch1_estimate_pi_quasirandom.py
│   └── ch1_simple_random_walk.py
│
├── chapter_2/
│   ├── ch2_random_walk.py
│   └── ch2_prng_sets_A_B_C.py
...
├── chapter_8/
│   ├── ch2_aloha.py
│   ├── ...
│
└── README.md
└── README_docker.md
└── requirements.txt
└── environment.yml
└── environment_full.yml



```
 
## Running Scripts and Reproducing Results

To run a script, navigate to the corresponding chapter directory. For example, in Chapter 5, run:

```bash
cd chapter_5
python ch5_strat_sampling_pi.py
```

When run without arguments, each script uses the same parameters and  seed as in the book, so it reproduces the reported results and figures.

The output is typically displayed on screen, and figures are also saved as `.pdf` files in a subdirectory called `results/`. This folder is created automatically if it does not exist.

Most scripts support command-line arguments that allow you to explore different settings.  To see the available options use the `--help` flag, e.g., run:

```bash
python ch5_strat_sampling_pi.py --help
```

You can override the default values using flags.  

**Example:** the following command estimates $\pi$ using stratified sampling with 500 replications, 4 strata, and a custom seed:

```bash
python ch5_strat_sampling_pi.py --R 500 --m 4 --seed 54321
```

Many scripts also accept **comma-separated lists of values**.  For example, the default invocation shown above:

```bash
python ch5_strat_sampling_pi.py
```

is equivalent to the following:

```bash
python ch5_strat_sampling_pi.py --R 200,10000 --Rpilot 100 --m 5,10,20 --results_path results --seed 31415
```

This command estimates $\pi$ several times: using 200 and 10000 replications, and 5, 10, or 20 strata.

 

##   Full reproducibility with Docker (Optional)

In rare cases, if installing with Conda does not fully reproduce the expected results (e.g., due to subtle version differences or system dependencies), you can run all code examples inside a fully configured Docker container.

See [README_docker.md](README_docker.md) for full instructions.
 
## License
This code is provided under the MIT License. See the [LICENSE](LICENSE) file for details.



<br>

<br>

![Views](https://komarev.com/ghpvc/?username=lorek&repo=Lectures_on_Monte_Carlo_Theory&label=Repo%20views)
