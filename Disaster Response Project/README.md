# Disaster Response Pipeline Project

## Overview
In this project, I applied some data ETL data engineering skills to analyze disaster data from [Figure Eight](https://www.figure-eight.com/). The data contained real messages that were sent during disaster events and this project covered creating a machine learning pipeline to categorize these events so that classified messages can be routed to the appropriate disaster relief agency. The project included building an interface (web application) where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the training data.


## Project Structure
Disaster Response Main Folder
   |--app
   |   |--templates
   |   |    |--master.html #home html page
   |   |    |--go.html #message result response html page
   |   |--run.py  #main flask file
   | 
   |--data
   |   |--disaster_categories.csv #text data to process
   |   |--disaster_messages.csv  #text data to process
   |   |--DisasterResponse.db  #the database save
   |   |--process_data.py  #python file for data processing & cleaning
   |
   |--models
   |   |--classifier.pkl.csv #saved classifier of the ml model
   |   |--train_classifier.py  #python script to train model on data
   |
   |--README.md
   |--requirements.txt


## Process Descriptions


### ETL Pipeline

### ML Pipeline

### Flask Web Application



### Instructions on running the project:
1. Run the following commands in the project's root directory to set up the database and classifier model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves to disk
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
