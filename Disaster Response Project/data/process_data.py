import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from os import path
import re

sys.path.insert(1, '../data/')

def is_path(filepath, checktype='dir'):
    """Checks if a path or directory exists. 
    
    Args:
    filepath str: A string representing a path or directory.
    checktype str:  A string. Accepts values 'dir' for directory and 'file' for file. Default: 'dir'
    
    Returns:
    A boolean value: true if the path or directory exists and false otherwise.
     
    """
    if checktype == 'dir':
        if not path.isdir(filepath):
            print('WARNING: Save Path does not exist.')
            return False
    if checktype == 'file':
        if not path.isfile(filepath):
            print('WARNING: File path does not exist.')
            return False
    return True

def check_inputs(inputs, file_types):
    """Checks if multiple inputs exist as files or directories. Uses the is_path function. 
    
    Args:
    inputs list or array of strings: Contains all the directories to check.
    file_types list or array of strings: Contains the expected file type for each input. Each list value should be a string of value 'file' or 'dir'
    
    Returns:
    A boolean value: true only if the all path or directories exist and false otherwise.
     
    """
    for i, filepath in enumerate(inputs):
        if not is_path(filepath, file_types[i]):
            return False
    return True

def clean_title(input_list):
    """Takes in an input list of string text. Clean strings and returns a list of capitalized strings
    
    Args:
    inputs list or array of strings: Contains all strings to reformat.

    Returns:
    A list value: A list of strings
    """
    return [re.sub(r'[^A-Za-z0-9]',' ',text).title() for text in input_list]


def load_data(messages_filepath, categories_filepath):
    """Loads csv data and creates a dataframes from filepaths.
    
    Args:
    messages_filepath str: The file path to the messages csv file
    categories_filepath str: The file path to the categories csv file
    
    Returns:
    A pandas dataframe of the merged csv files.
    
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    return pd.merge(messages_df, categories_df)


def clean_data(df):
    """Cleans a dataframe
    
    Args:
    df pandas.Dataframe: A pandas dataframe to clean
    
    Returns:
    A cleaned pandas dataframe.
    """
    #creates a dataframe of the individual category columns and renames them
    categories_df = df['categories'].str.split(';', expand=True)
    first_row = categories_df.iloc[0]
    categories_df.columns = first_row.apply(lambda x: x[:-2].strip())
    
    for column in categories_df:
        #cast as 0 or 1 and then from column values from string to numeric
        categories_df[column] = categories_df[column].apply(lambda x: x.split('-')[1])
        categories_df[column] = pd.to_numeric(categories_df[column])
    
    df.drop(columns=['categories'], inplace=True)
    
    clean_df = pd.concat([df,categories_df], sort=False, axis=1)
    clean_df.drop_duplicates(inplace=True)
        
    return clean_df


def save_data(df, database_filename):
    """Save content of a dataframe to a database
    
    Args:
    df pandas.Dataframe: A pandas dataframe to save
    database_filename str: A filepath for the database name
    
    Returns:
    A cleaned pandas dataframe.
    """
    database_filename = 'sqlite:///'+database_filename
    engine = create_engine(database_filename)
    df.to_sql(name="messages", con=engine, index=False, if_exists='replace')


def main():
    
    inputs = sys.argv
    file_types = ['file','file']
    
    if (len(inputs) == 4) and check_inputs(inputs[1:-1], file_types):
        
        messages_filepath, categories_filepath, database_filepath = inputs[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()