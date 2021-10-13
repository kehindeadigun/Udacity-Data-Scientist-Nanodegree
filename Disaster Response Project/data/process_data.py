import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load csv data and create a dataframes from filepaths.
    
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
    database_filename str: A filename for the database name
    
    Returns:
    A cleaned pandas dataframe.
    """
    if database_filename[-3:] != '.db':
        database_filename = database_filename+'.db'
    database_filename = 'sqlite:///'+database_filename
    engine = create_engine(database)
    df.to_sql(df, engine, index=False, if_exists='replace')
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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