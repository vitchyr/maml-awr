## Prerequisites

This project requires Python 3.6; Ubuntu 16.04 users can install it from Felix Kurll's deadsnakes PPA by following the instructions [here](https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get).

Required Python packages can be installed with `pip install -r requirements.txt`.

## Running training

Training can be invoked by running `python -m run`, which will run on the CPU by default. Training on a different device can be invoked by passing a device argument: `python -m run --device cuda:0` to run on the first GPU, for example.
