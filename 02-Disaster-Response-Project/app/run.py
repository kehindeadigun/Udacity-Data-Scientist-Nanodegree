import json
import plotly
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine

import re

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

#minor data cleaning function
def clean_title(input_list):
    """Takes in an input list of string text. Clean strings and returns a list of capitalized strings
    
    Args:
    inputs list or array of strings: Contains all strings to reformat.

    Returns:
    A list value: A list of strings
    """
    return [re.sub(r'[^A-Za-z0-9]',' ',text).title() for text in input_list]

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_dist = df.drop(columns=['id','message','original','genre']).sum(axis=0).reset_index()
    category_dist = category_dist.rename(columns={0:'values', 'index':'columns'})
    category_dist = category_dist.sort_values('values', ascending=False)
    
    category_names = clean_title(list(category_dist['columns']))
    category_counts = list(category_dist['values'])

    sorted_category_names = category_names[:8]
    sorted_category_counts = category_counts[:8]
    
    #Summary Category data for donut chart
    others_sum = sum(category_counts[8:])
    sorted_category_names.insert(0, 'Others')
    sorted_category_counts.insert(0, others_sum)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    marker_params = {
                        'color': '#9ECAE1',
                        'opacity': 0.6,
                        'line': {
                            'color': '#08306B',
                            'width': 1.5
                    }}
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=marker_params
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names[:5],
                    y=category_counts[:5],
                    marker=marker_params
                )
            ],

            'layout': {
                'title': 'Top Message Categories In Training Dataset',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=sorted_category_names,
                    values=sorted_category_counts,
                    hole= .4,
                    sort=True,
                    textposition='inside', 
                    textinfo='percent+label',
                    insidetextfont=dict(size=11),
                    insidetextorientation='radial',
                    marker={
                        'colors': ['#9ce3f0','#CFE5F0','#F7C59F','#6CC18D','#E37684','#70A3D0','#4D335B','#006BA6', '#D65780'],
                        'line': {
                            'color': '#08306B',
                            'width': 1.9,}
                            }
                    )
            ],
            'layout': {
                'title': 'Message Categories In Training Dataset'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()