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

#img = io.imread('/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/U3983_A.jpg')
#fig = px.imshow(img)
#fig.show()



image_filename = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/U3983_A.jpg' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())




image_filename2 = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/U3983_A.jpg' # replace with your own image
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())

#image_filename3 = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/U3983_B.jpg' # replace with your own image
#encoded_image3 = base64.b64encode(open(image_filename2, 'rb').read())

######################################################################################################

path = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/' 


flist = list()
for file in os.listdir(path):
    data = dict()
    base = os.path.basename(path + file)
    data["label"] = base
    open_file = open(path + file,'rb')
    image_read = open_file.read()
    image_64_encode = base64.encodebytes(image_read)
    data["data"] = image_64_encode.decode('ascii')
    flist.append(data)     

    final_data = json.dumps({'files':flist}, sort_keys=True, indent=4, separators=(',', ': '))
    #return render_template("images.html", final_data=final_data)


final_data[1]


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







app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

   
    
    
    


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
                                    'height': '200px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px',
                                    "margin-left": "28px",
                                },
                                # Allow multiple files to be uploaded
                                multiple=True
                            ),], style={"width": "18rem", 
                                         "margin-left": "50px",
                                         }),
                        width = 3
                        ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H1("Sketch", className="card-text")),
                                html.Div(id='output-image-upload', style = {'textAlign': 'center',
                                                                            'width': '80%',
                                                                            'height': '615px',
                                                                            'lineHeight': '60px',
                                                                            #'borderWidth': '5px',
                                                                            #'borderStyle': 'solid',
                                                                            #'borderRadius': '5px',
                                                                            'margin-left': '40px'}), dbc.CardFooter(" ")]), width = 3),               
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H1("Browse Matches", className="card-text")),
                                html.Div([
                                    dbc.Row(
                                        [
                                            dbc.Col(dbc.Button("Left", id="left_click", color="primary", className="mr-2"), style = {'textAlign': 'left'}),
                                            dbc.Col(html.Span(id="example-output2", style={"vertical-align": "middle"})),
                                            dbc.Col(dbc.Button("Right", id="right_click", color="primary", className="mr-2"), style = {'textAlign': 'right'}),
                                        ]),], style = {'textAlign': 'center',
                                                       'width': '80%',
                                                       'height': '550px',
                                                       'margin': '40px',}),
                                                dbc.CardFooter("Probability: 98%, Matches: 35"),
                                ], style={"width": "50rem"}),  width = 6),
                            ]
                        ),
                    ]
                )



                                                



def parse_contents(contents, filename, date):
    string = contents.split(";base64,")[-1]
    im_pil = drc.b64_to_pil(string)
    im_pil = im_pil.filter(ImageFilter.EDGE_ENHANCE_MORE)
    #buff = _BytesIO()
    #im_pil.save(buff, format="png")
    #im_bytes = im_pil.tobytes()
    #encoding_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    #encoding_string = drc.pil_to_b64(im_pil)
    return html.Div([
        html.H5("Manatee ID:     " + filename[0:7]),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        #drc.InteractiveImagePIL(
        #    image_id="interactive-image",
        #    image=im_pil,
            #enc_format=enc_format,
            #dragmode=dragmode,
            #verbose=DEBUG,
        #),
        html.Img(src=contents),
        #html.Hr(),
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
    return html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))






@app.callback(
    Output("example-output2", "children"), [Input("right_click", "n_clicks")]
)
def on_button_click(n):
    print(n)
    if n is None:
        return html.Img(src='data:image/png;base64,{}'.format(blank.decode()))
    else:
        return return_image(n)


@app.callback(
    Output("example-output", "children"), [Input("left_click", "n_clicks")]
)
def on_button_click2(n):
    global clicks
    if n is None:
        return " "
    else:
        return html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))











if __name__ == '__main__':
    app.run_server()










