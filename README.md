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
Build and deploy a model that predicts dataset's minority class - Female Character

## Data preparation and feature matrix
All EDA and ML models trainings you can find in notebook.ipynb

## How to run the project
`git clone https://github.com/alex-volosha/marvel-character-prediction`

# Run the app as a web service locally
* Build docker image\
(No need to run pipenv, Dockerfile will do it itself)\
by running this command in your terminal
`$ docker build -t  marvel .`

* Run the docker image
`$ docker run -it --rm -p 9696:9696 marvel`

* Open new terminal window and run prediction script
`$ python character.py`
And you will get the prediction of character being female. 

* As it's shown in features importance chart, two of the most important features are eyes, and hair. 
For example, by changing variable in character.py file from:\
'hair' : 'Red Hair'\
to:\
'hair' : 'Bald'\
The probability of the character being Female drops drastically, and therefor treashhold isn' reached, so as a result we get Female: 'False'

