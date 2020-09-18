#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:16:08 2020

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
from Manatee_Dashboard import dash_reusable_componets as drc
from PIL import ImageFilter  
import base64
from io import BytesIO
import os
import json
import numpy as np
from dash_canvas import DashCanvas
import dash_daq as daq
from dash_canvas.utils import array_to_data_url, parse_jsonstring, segmentation_generic
from dash.exceptions import PreventUpdate
from dash_canvas.utils.io_utils import image_string_to_PILImage
import torchvision
import torchvision.transforms as transforms                         
import torch
import torch.nn as nn
import operator
import torch.nn.functional as F
import skimage.measure
import math
from skimage.morphology import skeletonize

######################################################################################################


def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/index")),
            dbc.NavItem(dbc.NavLink("Database", href="/index")),           
        ],
        brand="Manatee Identification",
        color="primary",
        dark=True,
    )
    return navbar




path = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/' 

num_returned = 0
name_info = None
images = []
names = []
for image in os.listdir(path):
    #print("loading image: " + image)
    im = cv2.imread(path + image)
    name = image
    images.append(im)
    names.append(name)
    
    
names = np.array(names)  


from PIL import Image
im_pil = Image.fromarray(images[1])
im_bytes = im_pil.tobytes()
encoding_string = base64.b64encode(im_bytes).decode("ascii")





class Compare_ROIS(object):
    def __init__(self, path, input_sketch, roi, mask):
        self.path = path
        self.mask = mask
        self.input_sketch = input_sketch
        self.roi = roi #[x1,y1,x2,y2]
        self.processed_images = None
    def compare_rois(self):
        # get ROI array
        input_sketch_roi = self.input_sketch[int(self.roi[0]): int(self.roi[1]), int(self.roi[2]): int(self.roi[3])]
        # preprocess input sketch roi
        input_roi = self.preprocess(input_sketch_roi)        
        # find contours in input sketch roi
        input_contours = cv2.findContours(input_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # find contour rois in input sketch roi
        input_contours_shapes, input_contour_area, input_num_contours, input_bb_dims = self.find_contours(input_contours[0], input_roi)
        #Image.fromarray(input_contours_shapes[0]).show()
        #output list
        distance_dict = []
    # First get all file names in list file_names
        for i in range(len(self.processed_images)):
            # get ROI array and preprocess
            sketch_roi = self.processed_images[i][1][int(self.roi[0]): int(self.roi[1]), int(self.roi[2]): int(self.roi[3])]
            # find contours in ROI
            contours = cv2.findContours(sketch_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # get contours rois in sketch roi
            contours_shapes, contour_area, num_contours, bb_dims  = self.find_contours(contours[0], sketch_roi)
            try:                
                distances = self.compute_distances(input_bb_dims, bb_dims)
                if distances != "NA":
                    distance_dict.append((str(self.processed_images[i][0]), distances, input_roi, sketch_roi, input_contours_shapes, contours_shapes, input_contour_area, contour_area))        
            except:
                continue        

        distance_dict = sorted(distance_dict, key = lambda x: x[1]) 
        return distance_dict
    def preprocess(self, img):
        # black background
        img = cv2.bitwise_not(img)
        # threshold
        _,img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        # blur
        img = cv2.blur(img, (2,2))
        return img
    def find_contours(self, contours_list, sketch_roi):
        contours_rois = []
        contour_area = []
        num_contours = 0
        bb_dims = []
        for contour in contours_list:
            #filled = cv2.fillPoly(sketch_roi.copy(), contours_list, 255)           
            x, y, w, h = cv2.boundingRect(contour)
            roi = sketch_roi[y:y + h, x:x + w]                    
            area = cv2.contourArea(contour)          
            if area > 20 and len(contour) >= 5:
                (x,y), (MA,ma), angle = cv2.fitEllipse(contour)
                contours_rois.append(roi)
                contour_area.append(area)
                bb_dims.append([np.array([w,h]), area, angle, np.array([MA,ma])]) 
                num_contours += 1
        return contours_rois, contour_area, num_contours, bb_dims
    def compute_distances(self, input_contours_shape, contours_shape):
        distances = []
        num_input_scars = len(input_contours_shape)
        num_scars = len(contours_shape)
        if num_input_scars <= num_scars and num_scars < num_input_scars+3:
            comparisons = []
            for shape in input_contours_shape:                
                for shape2 in contours_shape:
                    # Separate h,w and MA,ma
                    input_h, input_w = shape[0]
                    h,w = shape2[0]
                    input_MA, input_ma = shape[3]  
                    MA, ma = shape2[3]                                        
                    
                    # Compute percentage differences for each feature
                    diff_in_MA = abs(input_MA - MA)
                    percentage_MA = (100*diff_in_MA)/input_MA
                    diff_in_ma = abs(input_ma - ma)/ input_ma
                    percentage_ma = (100*diff_in_ma)/input_ma
                    #diff_in_h = abs(input_h - h)
                    #percentage_h = (100*diff_in_h)/input_h
                    #diff_in_w = abs(input_w - w)
                    #percentage_w = (100*diff_in_w)/input_w
                    diff_in_area = abs(shape[1] - shape2[1])
                    percentage_area = (100*(diff_in_area))/shape[1]
                    diff_in_angle = abs(shape[2] - shape2[2])
                    percentage_angle = (100*(diff_in_angle))/shape[2]
                    comparisons.append(np.mean([percentage_area, percentage_angle, percentage_MA, percentage_ma]))
                    print(comparisons)
                if len(comparisons) != 0:
                    idx = np.argmin(comparisons)
                    best_match = comparisons[idx]
                    distances.append(best_match)
            return distances[0]
        else:
            return "NA"
    def removeOutline(self, img, mask):
        mask = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
        blur = cv2.GaussianBlur(mask,(5,5),0)
        mask = cv2.addWeighted(blur,1.5,mask,-0.5,0)
        mask[mask != 0] = 1          
        img = cv2.resize(img, (259, 559), interpolation= cv2.INTER_NEAREST)
        img[mask == 1] = 255
        return img
    def preLoadData(self):
        sketch_names = [] 
        processed_images = []
        for file_ in sorted(os.listdir(self.path)):
            if file_[-1] == 'g':
                sketch_names.append(str(file_))
            if file_[-1] == 'G':
                sketch_names.append(str(file_))
        for i in range(len(sketch_names)):
            # get sketch path
            sketch_path = str(self.path + str(sketch_names[i]))
            # read sketch in grayscale format
            sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
            sketch_no_outline = self.removeOutline(sketch, self.mask)
            preprocessed_img = self.preprocess(sketch_no_outline)
            processed_images.append([str(sketch_names[i]), preprocessed_img])
        self.processed_images = processed_images


path_to_images = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/'
path_to_mask = '/Users/natewagner/Documents/Mote_Manatee_Project/canny_filled2.png'

# initiate class:
find_matches_func = Compare_ROIS(path_to_images, None, None, path_to_mask)
find_matches_func.preLoadData()
Image.fromarray(find_matches_func.processed_images[0][1])




def getRowColIndex(matrix):
    row_num = 0
    col_num = 0
    
    for i in matrix:
        if (i == 0).all():
            row_num += 1
        else:
            break
    
    for i in matrix.T:
        if (i == 0).all():
            col_num += 1
        else:
            break
    return [row_num, col_num]


######################################################################################################




img_size = 100






width = 259
height = 559
new_matches = None

        











app = dash.Dash(external_stylesheets=[dbc.themes.LITERA], assets_folder='/Users/natewagner/Documents/Mote_Manatee_Project/Manatee_Dashboard/assets/')

filename = app.get_asset_url("BLANK_SKETCH_updated.jpg")



blank = os.getcwd() + '/Mote_Manatee_Project/BLANK_SKETCH_updated.jpg'
canvas_width = 259



app.layout = html.Div(
    [
     Navbar(),
     html.Br(),
     html.Br(),
     dbc.Row(
            [
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H1("Sketch", className="card-text"), style={"width": "50rem", 'textAlign': 'center'}),
                        dbc.Row([
                        dbc.Col(
                            html.Div([
                                DashCanvas(
                                    id='canvas',
                                    width=canvas_width,
                                    filename=filename,
                                    hide_buttons=['line', 'select'],
                                    goButtonTitle="Enter"
                                    )
                                ], className="six columns", style = {"margin-left": "50px"}), width = 8),
                        dbc.Col(
                            dbc.Row([
                            dbc.Card([
                                dbc.CardHeader(html.H4("Toolbox", className="card-text"), style={'textAlign': 'center'}),                                
                                html.Div([
                                    html.H6(children=['Brush Width'], style={'textAlign': 'center', 'font-weight': 'normal'}),
                                    dcc.Slider(
                                        id='bg-width-slider',
                                        min=1,
                                        max=40,
                                        step=1,
                                        value=1
                                    ),     
                                    html.H6(children=['Brush Color'], style={'textAlign': 'center', 'font-weight': 'normal'}),
                                    dcc.RadioItems(
                                            id='color-picker',
                                            options=[
                                                {'label': 'Black', 'value': dict(hex='#000000')},
                                                {'label': 'Grey', 'value': dict(hex='#666666')},
                                            ],
                                            value=dict(hex='#000000'),
                                            labelStyle={'display': 'inline-block', 'margin-right': '20px', 'margin-left': '20px', 'font-weight': 300},
                                            inputStyle={"margin-right": "10px"},
                                            style={'textAlign': 'center', 'font-weight': 'normal', 'font-size' : '15'}
                                        )  
                        ], style = {"margin-top": "10px"}, className="three columns")], style = {'width': '14.4rem', 'margin-top': '80px'}),
                        dbc.Card([
                            dbc.CardHeader(html.H4("Filters", className="card-text"), style={'textAlign': 'center'}), 
                                    html.H6(children=['Region'], style={'textAlign': 'center', 'font-weight': 'normal'}),
                                    dcc.Checklist(
                                        options=[
                                            {'label': 'New York City', 'value': 'NYC'},
                                            {'label': 'Montréal', 'value': 'MTL'},
                                            {'label': 'San Francisco', 'value': 'SF'}
                                        ],
                                        value=['NYC', 'MTL'],
                                            labelStyle={'display': 'inline-block', 'margin-right': '20px', 'margin-left': '20px', 'font-weight': 300},
                                            inputStyle={"margin-right": "10px"},
                                            style={'textAlign': 'left', 'font-weight': 'normal', 'font-size' : '15'}
                                    )                          
                            
                            ], style = {'width': '14.4rem', 'margin-top': '80px'})
                        
                        ]), width = 4)]),
                        
                        
                    ], style = {'align-items': 'center', 'width': '50rem', 'margin-left': '40px'}), width = 6),               
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H1("Browse Matches", className="card-text", style = {'textAlign': 'center'}), style={"width": "50rem"}),
                                html.Div([
                                    dbc.Row(
                                        [
                                            dbc.Col(dbc.Row([dbc.Button("Left", id="left_click", color="primary", className="mr-2", n_clicks = 0, size="lg")]), style = {'Align': 'left', 'vertical-align': 'middle'}, width = 2),
                                            dbc.Col(html.Span(id="sketch_output", style={"vertical-align": "middle"}), width = 8),
                                            dbc.Col(dbc.Row([dbc.Button("Right", id="right_click", color="primary", className="mr-2", n_clicks = 0, size="lg")]), style = {'Align': 'right', 'vertical-align': 'middle'}, width = 2),
                                        ]),], style = {'textAlign': 'center',
                                                       'width': '80%',
                                                       'height': '600px',
                                                       'margin': '40px',
                                                       'align-items': 'center',
                                                       'margin-top': '20px'}),
                                                dbc.CardFooter(), #"Probability: 98%, Matches: 35", style={"width": "50rem"}
                                ], style={"width": "50rem",
                                          "align-items": "center"}),  width = 6),
                            ]
                        ),
                    ]
                )



                                                



def parse_contents(contents, filename, date):
    string = contents.split(";base64,")[-1]
    decoded = base64.b64decode(string)
    buffer = BytesIO(decoded)
    im_pil = Image.open(buffer)
    #test = drc.DisplayImagePIL("Manatee ID", im_pil)
    return html.Div([
        html.Br(),
        html.H5("Manatee ID:     " + filename[0:-4]),
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
blank = base64.b64encode(open('/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated.jpg', 'rb').read())

def return_image(n):
    image_filename2 = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/'
    global names
    global num_returned
    file = names[n]
    
    encoded_image = base64.b64encode(open(image_filename2 + file, 'rb').read())
    if name_info is None:
        return html.Div([
        html.H5("Manatee ID: " + names[n][0:-4]),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
        ], style = {'align-items': 'center'})
    else:
        return html.Div([
            html.H5("Manatee ID: " + names[n][0:-4] + "   " + "Score: " + str(round(name_info[n], 3)) + "  Matches: " + str(num_returned)),
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
            ], style = {'align-items': 'center'})





count = 0
count2 = 0
switch = False

@app.callback(
    Output("sketch_output", "children"),
    [
     Input("right_click", "n_clicks"),
     Input("left_click", "n_clicks"),
     Input('canvas', 'json_data')
     ]
)
def on_button_click(n, n2, run):
    global count
    global switch
    global names
    if switch == True:
        return_image(0)        
        count = 0
        switch = False   
        
    if count == len(names) - 1:
        count = 0
    #print("N is ", n2, "N2 is ", n)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    #print(changed_id)
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



@app.callback(Output('canvas', 'lineColor'),
            [Input('color-picker', 'value')])
def update_canvas_linecolor(value):
    if isinstance(value, dict):
        return value['hex']
    else:
        return value


@app.callback(Output('canvas', 'lineWidth'),
            [Input('bg-width-slider', 'value')])
def update_canvas_linewidth(value):
    return value


t = None

@app.callback(Output('sketch_output', 'data'),
                [Input('canvas', 'json_data')],
                [State('canvas', 'image_content')])
def update_data(string, image):    
    global new_matches
    global name_info
    global names
    global switch
    global count
    global find_matches_func
    global num_returned
    global t
    
    switch = True
    is_rect = False
    if string:
        data = json.loads(string)
        bb_info = data['objects'][1:]  
        #print(bb_info)        
 
        bounding_box_list = []
        
        for i in bb_info:
            if i['type'] == 'rect':  
                is_rect = True
                top = i['top']
                left = i['left']
                wd = i['width']
                ht = i['height']
                bounding_box_list.append((top, top+ht, left, left+wd))
            else:
                continue
 
        if is_rect == False:
            bounding_box_list.append((0, 559, 0, 259))
        
        mask = parse_jsonstring(string, shape=(height, width))
        mask = (~mask.astype(bool)).astype(int)
        mask[mask == 1] = 255
        mask = mask.astype(np.uint8)

        #cropped_mask = mask[int(bounding_box[0]): int(bounding_box[1]), int(bounding_box[2]): int(bounding_box[3])]
        
        find_matches_func.input_sketch = mask
        find_matches_func.roi = bounding_box_list[0]
        matches = find_matches_func.compare_rois()
        t = matches
        is_rect = False
        
       # matches = find_matches(cropped_mask)
        
        names1 = []
        for i in matches:
            names1.append(i[0])
        names = names1    
        
        new_matches = matches
        
        name_info = [i[1] for i in matches]
        num_returned = len(matches)
        #image_string = array_to_data_url((255 * new_sketch).astype(np.uint8))
        

        
    return #array_to_data_url((255 * data).astype(np.uint8))





if __name__ == '__main__':
    app.run_server()










