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
from Manatee_Dashboard import dash_reusable_componets as drc
from PIL import ImageFilter  
import base64
from io import BytesIO as _BytesIO
import os
import json
import numpy as np



######################################################################################################

path = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/' 

images = []
names = []
for image in os.listdir(path):
    print("loading image: " + image)
    im = cv2.imread(path + image)
    name = image
    images.append(im)
    names.append(name)
names = np.array(names)  


from PIL import Image
im_pil = Image.fromarray(images[1])
im_bytes = im_pil.tobytes()
encoding_string = base64.b64encode(im_bytes).decode("ascii")


######################################################################################################





app = dash.Dash(external_stylesheets=[dbc.themes.YETI])

   
    
    


app.layout = html.Div(
    [
     navbar.Navbar(),
     html.Br(),
     html.Br(),
     dbc.Row(
            [
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H1("Photo Upload", className="card-text")),
                        dcc.Upload(
                                id='upload-image',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '80%',
                                    'height': '100px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px',
                                    "margin-left": "28px",
                                },
                                multiple=True
                            ),], style={"width": "18rem", 
                                         "margin-left": "50px",
                                         }),
                        width = 3
                        ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H1("Sketch", className="card-text"), style={"width": "25rem", 'textAlign': 'center'}),
                                html.Div(id='output-image-upload', style = {'textAlign': 'center',
                                                                            'height': '650px',
                                                                            'align-items': 'center'}), 
                                    dbc.CardFooter(
                                        html.Div(
                                            [
                                                dbc.Button("Update Sketch", id="open", color = "primary"),
                                                dbc.Modal(
                                                    [
                                                        dbc.ModalHeader("WARNING"),
                                                        #dbc.Row([
                                                        dbc.ModalBody("Are you sure you want to update the sketch?"),
                                                            dbc.Button("Update", id="close", className="ml-auto", color = "danger")   
                                                        #    ])                                                   
                                                    ],
                                                    id="modal",
                                                ),
                                            ]
                                        ), style = {"width": "25rem"})], style = {'align-items': 'center',
                                                                                  'width': '25rem'}), width = 3),               
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H1("Browse Matches", className="card-text", style = {'textAlign': 'center'}), style={"width": "50rem"}),
                                html.Div([
                                    dbc.Row(
                                        [
                                            dbc.Col(dbc.Row([dbc.Button("Left", id="left_click", color="primary", className="mr-2", n_clicks = 0, size="lg")]), style = {'Align': 'left', 'vertical-align': 'middle'}, width = 2),
                                            dbc.Col(html.Span(id="example-output2", style={"vertical-align": "middle"}), width = 8),
                                            dbc.Col(dbc.Row([dbc.Button("Right", id="right_click", color="primary", className="mr-2", n_clicks = 0, size="lg")]), style = {'Align': 'right', 'vertical-align': 'middle'}, width = 2),
                                        ]),], style = {'textAlign': 'center',
                                                       'width': '80%',
                                                       'height': '600px',
                                                       'margin': '40px',
                                                       'align-items': 'center',
                                                       'margin-top': '20px'}),
                                                dbc.CardFooter("Probability: 98%, Matches: 35", style={"width": "50rem"}),
                                ], style={"width": "50rem",
                                          "align-items": "center"}),  width = 6),
                            ]
                        ),
                    ]
                )



                                                



def parse_contents(contents, filename, date):
    string = contents.split(";base64,")[-1]
    im_pil = drc.b64_to_pil(string)
    im_pil = im_pil.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return html.Div([
        html.Br(),
        html.H5("Manatee ID:     " + filename[0:7]),
        html.Img(src=contents),
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






image_filename2 = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/' # replace with your own image
encoded_image2 = base64.b64encode(open(image_filename2 + names[0], 'rb').read())
blank = base64.b64encode(open(image_filename2[0:54] + "BLANK_SKETCH_updated.jpg", 'rb').read())

def return_image(n):
    image_filename2 = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/'
    global names
    file = names[n]
    
    encoded_image = base64.b64encode(open(image_filename2 + file, 'rb').read())
    return html.Div([
        html.H5("Manatee ID: " + names[n][0:7]),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
        ], style = {'align-items': 'center'})





count = 0
count2 = 0
switch = False

@app.callback(
    Output("example-output2", "children"),
    [
     Input("right_click", "n_clicks"),
     Input("left_click", "n_clicks")
     ]
)
def on_button_click(n, n2):
    global count
    global switch
    global count2
    print("N is ", n2, "N2 is ", n)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(changed_id)
    if "right_click" in changed_id:
        count += 1
        return return_image(count)
    if "left_click" in changed_id:
        count -= 1
        return return_image(count)
    else:
        return html.Img(src='data:image/png;base64,{}'.format(blank.decode()))
    
    
    
    
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(nn1, nn2, is_open):
    if nn1 or nn2:
        return not is_open
    return is_open








if __name__ == '__main__':
    app.run_server()








