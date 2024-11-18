# Marvel Characters

The dataset I used: https://www.kaggle.com/datasets/fivethirtyeight/fivethirtyeight-comic-characters-dataset?select=dc-wikia-data.csv 

The data comes from [Marvel Wikia](http://marvel.wikia.com/Main_Page) and [DC Wikia](http://dc.wikia.com/wiki/Main_Page).

It's split into two files, for DC and Marvel, respectively: `dc-wikia-data.csv` and `marvel-wikia-data.csv`. Each file has the following variables:

Variable | Definition
---|---------
`page_id` | The unique identifier for that characters page within the wikia
`name` | The name of the character
`urlslug` | The unique url within the wikia that takes you to the character
`ID` | The identity status of the character (Secret Identity, Public identity, [on marvel only: No Dual Identity])
`ALIGN` | If the character is Good, Bad or Neutral
`EYE` | Eye color of the character
`HAIR` | Hair color of the character
`SEX` | Sex of the character (e.g. Male, Female, etc.)
`GSM` | If the character is a gender or sexual minority (e.g. Homosexual characters, bisexual characters)
`ALIVE` | If the character is alive or deceased
`APPEARANCES` | The number of appareances of the character in comic books (as of Sep. 2, 2014. Number will become increasingly out of date as time goes on.)
`FIRST APPEARANCE` | The month and year of the character's first appearance in a comic book, if available
`YEAR` | The year of the character's first appearance in a comic book, if available

## Goal
Build and deploy a model that predicts dataset's minority class - "Female Character"

## Data preparation and feature matrix:
All EDA and ML models trainings you can find in `notebook.ipynb`

## Run the project:
`git clone https://github.com/alex-volosha/marvel-character-prediction`

## Run the app as a web service locally:
* After cloning the repo you can install virtual environment dedicated for this project with all dependancies.\
(Make sure to go to the project directory in you terminal before you run this):\
`pip install pipenv`

Then install Pipfile/Pipfile.lock files by:\
`pipenv install`

 Narrow down into newly created virtual environment:\
`pipenv shell`

And now you can run `python predict.py` script.\
Open a new terminal and send request to running predict.py script by calling `python character.py`

## Run the app locally within a Docker container
> :warning: **Warning:** First make sure Docker is installed and running so you can connect it.
[Check Docker website to Install Docker Engine](https://docs.docker.com/engine/install/)

* Build docker image\
(No need to run pipenv, Dockerfile will do it itself) by running this command in your terminal\
`docker build -t  marvel .`

* Run the docker image\
`docker run -it --rm -e PORT=9696 -p 9696:9696 marvel`

* Open a new terminal window and run the prediction script\
`python character.py`\
And you will get the prediction of whether the character is female or not. 

> :bulb: **Options:** As shown in the features importance chart, two of the most important features are eyes and hair.\
For example, by changing the variable in the `character.py` file from:\
'hair' : 'Red Hair'\
to:\
'hair' : 'Bald'\
The probability of the character being Female drops drastically, and therefore, the threshold isn't reached, so, as a result, we get Female: 'False'

