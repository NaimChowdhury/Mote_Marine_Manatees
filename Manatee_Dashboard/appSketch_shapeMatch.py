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


######################################################################################################


path = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/' 

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
        Image.fromarray(input_roi).show()
        # find contours in input sketch roi
        input_contours = cv2.findContours(input_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # find contour rois in input sketch roi
        input_contours_shapes, input_contour_area, input_num_contours = self.find_contours(input_contours[1], input_roi)
        #output list
        distance_dict = []
    # First get all file names in list file_names
        for i in range(len(self.processed_images)):
            # get ROI array and preprocess
            sketch_roi = self.processed_images[i][1][int(self.roi[0]): int(self.roi[1]), int(self.roi[2]): int(self.roi[3])]
            # find contours in ROI
            contours = cv2.findContours(sketch_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # get contours rois in sketch roi
            contours_shapes, contour_area, num_contours  = self.find_contours(contours[1], sketch_roi)
            distances = self.compute_distances(input_contours_shapes, contours_shapes)
            try:
                score = np.min(distances) + abs(input_num_contours - num_contours)*0.05 #+ abs(np.sum(input_contour_area) - np.sum(contour_area))
            except:
                score = 0
            if score != 0:
                distance_dict.append((str(self.processed_images[i][0]), score, sketch_roi, sketch_roi))        
        distance_dict = sorted(distance_dict, key = lambda x: x[1]) 
        return distance_dict
    def preprocess(self, img):
        # black background
        img = cv2.bitwise_not(img)
        # threshold
        _,img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        # blur
        #img[img != 0] = 1
        #skeleton = skeletonize(img, method = "lee")
        #skeleton[skeleton != 0] = 255
        img = cv2.blur(img, (2,2))
        return img
    def find_contours(self, contours_list, sketch_roi):
        contours_rois = []
        contour_area = []
        num_contours = 0
        for contour in contours_list:
            x, y, w, h = cv2.boundingRect(contour)
            roi = sketch_roi[y:y + h, x:x + w]
            area = cv2.contourArea(contour)
            if area > 20:
                contours_rois.append(roi)
                contour_area.append(area)
                num_contours += 1
        return contours_rois, contour_area, num_contours
    def compute_distances(self, input_contours_shape, contours_shape):
        distances = []
        for shape in input_contours_shape:
            for shape2 in contours_shape:
                distance = cv2.matchShapes(shape, shape2, cv2.CONTOURS_MATCH_I1,0)
                distances.append(distance)
        return distances
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



def get_pixel_matricies(path_to_images, path_to_mask):
    mask = cv2.imread(path_to_mask)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_mask,(5,5),0)
    mask = cv2.addWeighted(blur,1.5,gray_mask,-0.5,0)
    mask[mask != 0] = 1
    

    data = []
    for img in os.listdir(path_to_images):
        if img == ".DS_Store":
            continue
        else:
            image = cv2.imread(path_to_images + img)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            image = cv2.resize(image, (259, 559), interpolation= cv2.INTER_NEAREST)
            
            image[mask == 1] = 255
    
            image = cv2.resize(image, (100, 100), interpolation= cv2.INTER_NEAREST)
    
            #if img == "DUSW10501_B.jpg":
            #    Image.fromarray(image).show()
    
            tiles = []
            for i in range(0, 100, 10):
                for j in range(0, 100, 10):
                    roi = image[i:i+10, j:j+10]
                    tiles.append(roi)
                        
            pixels = []
            for x in tiles:
                pixels.append(len(x[x < 250]))
    
            pixels = np.matrix(pixels).reshape((10,10))
            #pixel_matrix = skimage.measure.block_reduce(pixels, (3,3), np.max)           
            
            data.append([img, pixels])

        
    return data
        
        

pixel_map_data = get_pixel_matricies(path_to_images, path_to_mask)


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



def find_matches(img1):
    out_list = []
    
    
    
    
    
    anc = cv2.resize(img1, (100, 100), interpolation= cv2.INTER_NEAREST)

    #Image.fromarray(anc).show()

    tiles = []
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            roi = anc[i:i+10, j:j+10]
            tiles.append(roi)

    means = []
    for x in tiles:
        means.append(len(x[x < 250]))
         

    means1 = np.matrix(means).reshape((10,10))

    #loc1 = getRowColIndex(means1)[0]
    #print(means1[loc1])
    #means1 = skimage.measure.block_reduce(means1, (3,3), np.max)

    


    for name,pixel_map in pixel_map_data:
        
        #loc2 = getRowColIndex(pixel_map)[0]
        
        
        score = abs(np.linalg.norm(means1-pixel_map))
        #score = abs(np.linalg.norm(means1-pixel_map))
                    
        if math.isnan(score) == False:
            out_list.append((name, score))
            
        dist_pairs = sorted(out_list, key = lambda x: x[1]) 
    
    return dist_pairs






width = 259
height = 559
new_matches = None

        











app = dash.Dash(external_stylesheets=[dbc.themes.LITERA], assets_folder='/Users/natewagner/Documents/Mote_Manatee_Project/Manatee_Dashboard/assets/')

filename = app.get_asset_url("BLANK_SKETCH_updated.jpg")



blank = os.getcwd() + '/Mote_Manatee_Project/BLANK_SKETCH_updated.jpg'
canvas_width = 259



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
                                ], className="six columns"), width = 8),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H4("Toolbox", className="card-text"), style={'textAlign': 'center'}),                                
                                html.Div([
                                    html.H6(children=['Brush width'], style={'textAlign': 'center', 'font-weight': 'normal'}),
                                    dcc.Slider(
                                        id='bg-width-slider',
                                        min=1,
                                        max=40,
                                        step=1,
                                        value=1
                                    ),
                                    daq.ColorPicker(
                                        id='color-picker',
                                        label='Brush color',
                                        value=dict(hex='#000000')
                            ),
                        ], style = {"margin-top": "10px"}, className="three columns")], style = {'width': '14.4rem', 'margin-top': '80px'}), width = 4)]),
                        
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
    im_pil = drc.b64_to_pil(string)
    im_pil = im_pil.filter(ImageFilter.EDGE_ENHANCE_MORE)
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
blank = base64.b64encode(open(image_filename2[0:54] + "BLANK_SKETCH_updated.jpg", 'rb').read())

def return_image(n):
    image_filename2 = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/'
    global names
    file = names[n]
    
    encoded_image = base64.b64encode(open(image_filename2 + file, 'rb').read())
    if name_info is None:
        return html.Div([
        html.H5("Manatee ID: " + names[n][0:-4]),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
        ], style = {'align-items': 'center'})
    else:
        return html.Div([
            html.H5("Manatee ID: " + names[n][0:-4] + "   " + "Prob: " + str(round(name_info[n], 3))),
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
    global t
    
    switch = True
    if string:
        data = json.loads(string)
        bb_info = data['objects'][1:]     
        try:
            for i in bb_info:
                if i['type'] == 'rect':         
                    top = i['top']
                    left = i['left']
                    wd = i['width']
                    ht = i['height']
                    bounding_box = ((top, top+ht, left, left+wd))
        except:
            print('No Bounding Box Found')
        
        
        mask = parse_jsonstring(string, shape=(height, width))
        mask = (~mask.astype(bool)).astype(int)
        mask[mask == 1] = 255
        mask = mask.astype(np.uint8)

        #cropped_mask = mask[int(bounding_box[0]): int(bounding_box[1]), int(bounding_box[2]): int(bounding_box[3])]
        
        find_matches_func.input_sketch = mask
        find_matches_func.roi = bounding_box
        matches = find_matches_func.compare_rois()
        t = matches
        
       # matches = find_matches(cropped_mask)
        
        names1 = []
        for i in matches:
            names1.append(i[0])
        names = names1    
        
        new_matches = matches
        
        name_info = [i[1] for i in matches]
        
        #image_string = array_to_data_url((255 * new_sketch).astype(np.uint8))
        

        
    return #array_to_data_url((255 * data).astype(np.uint8))





if __name__ == '__main__':
    app.run_server()








