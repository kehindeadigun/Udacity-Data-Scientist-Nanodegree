import sys

import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import joblib
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine
from os import path

sys.path.append('../data/')

from process_data import check_inputs

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """Loads data from a defined filepath
    
    Args:
    df pandas.Dataframe: A pandas dataframe to save
    database_filename str: A filename for the database name
    
    Returns:
    X numpy array: numpy Array: contains the message data values.
    Y numpy array: numpy Array: contains the message target values.
    category_names numpy Array:  contains the category name of each target value.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM messages',engine)
    target = df.drop(columns = ['id','message','original','genre'])
    
    X = df['message'].values
    Y = target.values
    category_names = np.array(target.columns)
    return X, Y, category_names


def tokenize(text):
    """A function to tokenize a text string. Strips the text of punctuation, stopwords and lemmatizes each word. 
    
    Args:
    text str: A string of text
    
    Returns: 
    A list of tokens or tokenized words.
    """
    clean_text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    tokens = word_tokenize(clean_text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word).lower().strip() for word in tokens]

def build_model():
    """
    Builds a multioutput text classifcation model. 
    
    Returns:
    A grid search multiclassification model with a randomforest estimator as base
    """
    
    vectorizer = CountVectorizer(tokenizer=tokenize, stop_words=stopwords.words("english"))
    transformer = TfidfTransformer()
    classifier = MultiOutputClassifier(RandomForestClassifier(verbose=1))
    
    pipeline = Pipeline([('vect', vectorizer),
                         ('tfidf', transformer),
                         ('clf', classifier)
                        ])
    
    parameters = { 'vect__ngram_range':[(1, 1),(1,2)], 'clf__estimator__n_estimators':list(range(1,30,5)) }
    
    model = GridSearchCV(pipeline, parameters, verbose=1)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model, printing out a classification report 
    
    Args:
    model: A scikit learn estimator
    X_test numpy array, list or dataframe: Contains test values from the data
    Y_test numpy array, list or dataframe: Contains test values from the target
    category_names numpy array or list: Contains the category name of each target value.
    """
    Y_pred = model.predict(X_test)
    for i in range(Y_pred.shape[1]):
        #for the first class, we have 3 labels and subsequent classes have 2 
        if i == 0:
            class_names = ['-'.join([str(category_names[i]), str(j)]) for j in range(3)]
        else:
            class_names = ['-'.join([str(category_names[i]), str(j)]) for j in range(2)]
        print('Column ',i+1,' : ',category_names[i])
        print(classification_report(Y_pred[:,i], Y_test[:,i], target_names=class_names))
        #print lines to visually separate categories in the output print
        print('-'*65)


def save_model(model, model_filepath=''):
    """Saves an ML model to the the filepath. Extracts and saves the best estimator if presented with a grid search model. Saves file name is 'classifier.pkl'.
    
    Args:
    model A model to save
    model_filepath str: File path to save the model to. (If filepath does not exist, will save to the model to the same folder as the train_classifier script).
    """
    try:
        best_model = model.best_estimator_
    except:
        best_model = model
    
    joblib.dump(best_model, model_filepath, compress=1)


def main():
    inputs = sys.argv
    if (len(inputs) == 3) and check_inputs(inputs[1:-1], ['file']):
        database_filepath, model_filepath = inputs[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl'\
              'and ascertain the database exists and the save path exists')

if __name__ == '__main__':
    main()