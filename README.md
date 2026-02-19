# sports-data-api
An API that returns sports statistics

# Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequistes)
- [Installation](#installation)
  - [Python Virtual Environment](#python-virtual-environment)
- [Getting Started](#getting-started)
  - [Spinning Up The Job Board API Container](#spinning-up-the-job-board-api-container)
  - [Running The Server Locally](#running-the-server-locally)
<<<<<<< HEAD
- [NFL Statistics](#nfl-statistics)
- [PGA Statistics](#pga-statistics)
- [MLB Statistics](#mlb-statistics)
- [Resources](#resources)
=======
- [NFL Statistics](nfl-statistics)
- [PGA Statistics](pga-statistics)
>>>>>>> 265c35e (Include additional stats and predictions for the pga_service)

# Introduction
This project utilizes machine learning concepts to return advanced statics to monitor a players' performance, trends, and predictions.

# Features
- **Ramdom Forest Classifier:** A non-linear model algorithm used to caluculate and monitor a players' performance.

# Prerequistes
- [Python 3.14](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Git](https://git-scm.com/)
- **For Windows (WSL):**
  - [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install)
  - [Ubuntu](https://documentation.ubuntu.com/wsl/latest/howto/install-ubuntu-wsl2/)
    - **Note:** If there is an issue starting the docker engine, you may need to add the user to the docker group. you can run this command:
      - `sudo usermod -aG docker $USER`

# Installation
## Python Virtual Environment
  - **Linux/WSL**
    - Open your WSL terminal and navigate to your project's directory.
    - Install the python virtual environment (if it doesn't exist).
      - `apt install python3.12-venv`
    - Create the virtual environment
      - `python3 -m venv my_env`
    - Activate the virtural environment
      - `source my_env/bin/activate`
    - Since the virtual environment is now activated, install the required packages for your environment.
      - `pip3 install package_name` or `pip3 install -r requirements.txt`
    - Deactivate the virtual environment when finished.
      - `deactivate`
  - **Windows(PowerShell):**
    - Open your Powershell terminal and navigate to your project's directory.
    - Install the python virtual environment (if it doesn't exist). Enter `python` in the terminal and a window should appear providing instructions to install python on your machine.
      - `python`
    - Create the virtual environment
      - `python -m venv my_venv`
    - Activate the virtural environment
      - `.\my_env\Scripts\Activate.ps1`
    - Since the virtual environment is now activated, install the required packages for your environment.
      - `pip install package_name` or `pip install -r requirements`
    - Deactivate the virtual environment when finished.
      - `deactivate`
  - **Mac:**
    - Navigate to your project's directory.
    - Create the virtual environment. Change "my_env" to your desired environment name.
      - `python3 -m venv my_env`
    - Activate the virtual environment.
      - `source my_env/bin/activate`
    - Since the virtual environment is now activated, install the required packages for your environment.
      - `pip install package_name` or `pip install -r requirements`
    - Deactivate the environment when finished
      - `deactivate`

# Getting Started

## Spinning Up The Sports Data API Container
The `env_template` file contains default variables to store credentials and other sensitive data. Create a `.env` file in your project and copy the environment variables from the template and store them into the new config file. Ensure the file is referenced in `.gitignore`.

### Running the Application
  - Start and build the application with the required docker command
    - `docker compose --env-file .env -f docker/docker-compose.yml up --build -d`

## Running the Server Locally
If you want to run the server locally to send API requests, you can use the command below and replace "main" with the name of your file that you want to run. In this application, the file we will want to run is app.py since the file contains the endpoints we want to retrieve data.
  - `python3 app.py`

You may run into an issue where the project uses the CMake command to find a package configuration file provided by "Arrow". The Apache Arrow dependency is required to to utilize functions within the pybaseball library. If this is the case, you may need to install the Apache Arrow library on your system.

  - For macOS:
    - `brew install apache-arrow`
  - For Ubuntu/Debian/Linux
    - `sudo apt-get update`
    - `sudo apt-get install -y libarrow-dev`

If the application continues to fail after installing the Apache Arrow library, then you can revert to running the application via docker compose.


# NFL Statistics
The nfl_services.py file contains a variety of methods that returns advanced statistics for monitoring player's and team's performances each week.

# PGA Statistics
The pga_services.py file contains metrics that predicts a player's golf performance at a given course. The logic includes a list of player's past performances from different tournaments using their Strokes Gained stats to determine future performance. The script also calculates a player's potential top 10 performance for future tournaments by comparing their current Strokes Gained stats against the world's talented players.

# MLB Statistics
The mlb_services.py file contains statistics that analyzes a pictcher's and batter's in-game performance.

**Key Metrics:**

  - ***wOBA (Weighted On-Base Average):*** This measures a player's overall offensive impact when a player reaches base. The forumula includes Formula includes walks, hit-by-pitch, singles, doubles, triples, and home runs.
  - ***xwOBA (Expected Weighted On-Base Average):*** It measures a player's offensive performance based on the quality of a ball being hit. The metric takes account of a ball's exit velocity and launch angle.
  - ***BABIP (Batting Average on Balls In Play):*** Determines how lucky or unlucky a player has been performing at bat. This metric can tell us how consistant a player has been getting hits over time.
  - ***WHIP (Walks plus Hits per Inning Pitched):*** This measures a pitcher's efficiency in preventing batters from reaching base.
  - ***Barrel %:*** Measures how frequently a batter hits a ball. This analyzes a player's exit velocity and launch angle.
  - ***ISO (Isolated Power):*** Measures a batter's raw power.
  - ***Platoon Stats:*** This anaylzes the handedness between a pitcher and batter for measuring splits to optimize matchups. Ex: left-hand pitcher vs. right-handed batter. 

# Resource
- [PGA Tour](https://www.pgatour.com/)
- [Course Rating and Slope Database](https://ncrdb.usga.org/courseTeeInfo?CourseID=14408)
- [Fantasy Data](https://fantasydata.com/)
- [Fantasy Pros](https://www.fantasypros.com/)
- [Odds Data](https://www.rotowire.com/betting/nfl/odds)
