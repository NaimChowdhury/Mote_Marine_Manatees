#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:02:28 2020

@author: natewagner
"""


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import cv2
import plotly.express as px
from skimage import io
from Manatee_Dashboard import navbar


img = io.imread('/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/U3983_A.jpg')
fig = px.imshow(img)
fig.show()






app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])



app.layout = html.Div(
    [
     navbar.Navbar(),
     html.Br(),
     html.Br(),
     dbc.Row([
         dbc.Col(html.H1("Photo Upload", style = {'padding-left': '1.5%', 'font-family' : 'Arial'}), width = 3),
         dbc.Col(html.H1("Sketch", style = {'font-family' : 'Arial'}), width = 3),
         dbc.Col(html.H1("Selected", style = {'font-family' : 'Arial'}), width = 3),
         dbc.Col(html.H1("Browse Matches", style = {'font-family' : 'Arial'}), width = 3),
     ]
     ),
     
        dbc.Row(
            [
                dbc.Col(
                        dcc.Upload(
                                id='upload-image',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '40%',
                                    'height': '200px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px',
                                    "margin-left": "50px",
                                },
                                # Allow multiple files to be uploaded
                                multiple=True
                            ),
                        width = 3
                        ),
                dbc.Col(html.Div(id='output-image-upload', style = {'textAlign': 'center',
                                                 'width': '80%',
                                                 'height': '615px',
                                                 'lineHeight': '60px',
                                                 'borderWidth': '5px',
                                                 'borderStyle': 'solid',
                                                 'borderRadius': '5px',
                                                 'margin': '10px'}), width = 3),
                dbc.Col(html.Div(id='output-image-upload2', style = {'textAlign': 'center',
                                                 'width': '80%',
                                                 'height': '615px',
                                                 'lineHeight': '60px',
                                                 'borderWidth': '5px',
                                                 'borderStyle': 'solid',
                                                 'borderRadius': '5px',
                                                 'margin': '10px'}), width = 3),
                dbc.Col(html.Div([
                        dcc.Graph(figure=fig)]), width = 3),
            ]
        ),
    ]
)






def parse_contents(contents, filename, date):
    return html.Div([
        html.H5("Manatee ID:     " + filename[0:7]),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
    ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



if __name__ == '__main__':
    app.run_server()










