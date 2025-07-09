##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##
##    --------------------  Dockerfile --------------------                 ##
##                                                                          ##
## Book: Lectures on Monte Carlo Theory                                     ##
## Authors: Paweł Lorek and Tomasz Rolski                                   ##
## Affiliation: University of Wrocław, Poland                               ##
##                                                                          ##
## Publisher: Springer, 2025                                                ##
## Springer link: link.springer.com/book/9783032011893                      ##
## Series: Probability Theory and  Stochastic Modelling                     ##
## ISBN:  978-3-032-01189-3                                                 ##
##                                                                          ##
## It uses the full Conda environment specification (environment_full.yml)  ##
## to ensure exact versions of all dependencies.                            ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##


FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /app

# Copy all repository contents into the container
COPY . /app/Lectures_on_Monte_Carlo_Theory

# Create the Conda environment from the full specification
RUN conda env create -f /app/Lectures_on_Monte_Carlo_Theory/environment_full.yml

# Ensure the environment is activated by default in future RUN/CMD
SHELL ["conda", "run", "-n", "MonteCarlo_book", "/bin/bash", "-c"]

# Set default command when container starts
CMD ["bash"]
