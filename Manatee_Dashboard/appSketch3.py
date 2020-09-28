

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import cv2
import base64
import os
import json
import numpy as np
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring
import pandas as pd
from PIL import Image
import dash_daq as daq


######################################################################################################


### Needed Paths ###
path_to_images = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MMLDUs_BatchA/'
path_to_mask = '/Users/natewagner/Documents/Mote_Manatee_Project/canny_filled2.png'
path_to_blank = '/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated.jpg'




def Navbar():
    navbar = dbc.NavbarSimple(
        children=[
            #dbc.NavItem(dbc.NavLink("Home", href="/index")),
            #dbc.NavItem(dbc.NavLink("Database", href="/index")),           
        ],
        brand="Manatee Identification",
        color="primary",
        expand = 'sm',
        dark=True,
    )
    return navbar




# get database and image info
num_returned = 0
name_info = None
images = []
names = []
for image in os.listdir(path_to_images):
    im = cv2.imread(path_to_images + image)
    name = image
    images.append(im)
    names.append(name)    
names = np.array(names)  






class Compare_ROIS(object):
    def __init__(self, path, input_sketch, roi, mask):
        self.path = path
        self.mask = mask
        self.input_sketch = input_sketch
        self.roi = roi #[x1,y1,x2,y2]
        self.processed_images = None
    def compare_rois(self):
        # get ROI array
        input_contour_info = []
        for input_bb in self.roi:  
            input_sketch_roi = self.input_sketch[int(input_bb[0]): int(input_bb[1]), int(input_bb[2]): int(input_bb[3])]
            # preprocess input sketch roi
            input_roi = self.preprocess(input_sketch_roi)        
            # find contours in input sketch roi
            input_contours = cv2.findContours(input_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # find contour rois in input sketch roi       
            input_shapes, input_area, input_num_contour, input_bb_dim = self.find_contours(input_contours[0], input_roi)
            input_contour_info.append([input_shapes, input_area, input_num_contour, input_bb_dim])
#            Image.fromarray(input_roi).show()

        distance_dict = []
    # First get all file names in list file_names
        for i in range(len(self.processed_images)):
            # get ROI array and preprocess     
            for x in range(len(self.roi)):
                sketch_roi = self.processed_images[i][1][int(self.roi[x][0]): int(self.roi[x][1]), int(self.roi[x][2]): int(self.roi[x][3])]
                # find contours in ROI
                contours = cv2.findContours(sketch_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # get contours rois in sketch roi                
                contours_shapes, contour_area, num_contours, bb_dims  = self.find_contours(contours[0], sketch_roi)
                #try:    
                distances = self.compute_distances(input_contour_info[x][3], bb_dims)  
                if distances != "NA":
                    distance_dict.append((str(self.processed_images[i][0]), distances))        
                #except:
                #    continue                
        distance_dict_df = pd.DataFrame(distance_dict)
        unique_names_cnts = distance_dict_df.groupby(0)[1].agg(['count', 'mean'])
        unique_names_cnts['names'] = unique_names_cnts.index
        has_all_scars = unique_names_cnts[unique_names_cnts['count'] >= len(self.roi)]
        returned = has_all_scars[['names', 'mean']]        
        returned_list = returned.values.tolist()        
        returned_list = sorted(returned_list, key = lambda x: x[1])         
        return returned_list
    def preprocess(self, img):
        # blur
        img = cv2.blur(img, (2,2))
        # black background
        img = cv2.bitwise_not(img)
        # threshold
        _,img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)        
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
            # aspect ratio
            aspect_ratio = float(w)/h                        
            # contour center coordinates
            contour_x = round(x + (w/2)) 
            contour_y = round(y + (h/2))               
            area = cv2.contourArea(contour)  
            # extent
            rect_area = w*h
            extent = float(area)/rect_area
            avg_pixel = np.sum(roi)
            if area > 20 and len(contour) >= 5:
                (x,y), (MA,ma), angle = cv2.fitEllipse(contour)
                contours_rois.append(roi)
                contour_area.append(area)
                bb_dims.append([np.array([w,h]), area, angle, np.array([MA,ma]), np.array([contour_x, contour_y]), avg_pixel, aspect_ratio, extent]) 
                num_contours += 1
        return contours_rois, contour_area, num_contours, bb_dims    
    def compute_distances(self, input_contours_shape, contours_shape):
        num_input_scars = len(input_contours_shape)
        num_scars = len(contours_shape)
        if num_input_scars == 0 and num_scars > 0:
            return 'NA'
        if num_input_scars == 0 and num_scars == 0:
            return 0
        #if num_input_scars != 0 and num_scars != 0:
        if num_input_scars <= num_scars and num_scars < num_input_scars+3:
            comparisons = []
            for shape in input_contours_shape:                
                for num, shape2 in enumerate(contours_shape):
                    # Separate h,w and MA,ma and x,y
                    input_h, input_w = shape[0]
                    h,w = shape2[0]
                    input_MA, input_ma = shape[3]  
                    MA, ma = shape2[3]
                    input_x, input_y = shape[4]
                    x,y = shape2[4]    
                    input_aspect = shape[6]
                    aspect = shape2[6]
                    input_extent = shape[7]
                    extent = shape2[7]                    
                    # Compute percentage differences for each feature
                    diff_in_x = abs(input_x - x)
                    percentage_in_x = (100*diff_in_x)/input_x
                    diff_in_y = abs(input_y - y)
                    percentage_in_y = (100*diff_in_y)/input_y
                    diff_in_MA = abs(input_MA - MA)
                    percentage_MA = (100*diff_in_MA)/input_MA
                    diff_in_ma = abs(input_ma - ma)/ input_ma
                    percentage_ma = (100*diff_in_ma)/input_ma
                    #diff_in_aspect = abs(input_aspect - aspect)/ input_aspect
                    #percentage_aspect = (100*diff_in_aspect)/input_aspect
                    #diff_in_extent = abs(input_extent - extent)/ input_extent
                    #percentage_extent = (100*diff_in_extent)/input_extent                    
                    #diff_in_pixs = abs(shape[5] - shape2[5])
                    #percentage_area = (100*(diff_in_pixs))/shape[5]
                    diff_in_angle = abs(shape[2] - shape2[2])
                    percentage_angle = (100*(diff_in_angle))/shape[2]
                    #comparisons.append(np.mean([percentage_area, percentage_angle, percentage_MA, percentage_ma, percentage_in_x, percentage_in_y]))
                    comparisons.append([num, 1/5*(0.10 * percentage_angle + 0.50 * percentage_MA + 0.30 * percentage_ma + 0.05 * percentage_in_x + 0.05 * percentage_in_y)])
            if len(comparisons) != 0:
                distances = self.computeScore(comparisons, num_input_scars)                                    
            return np.mean(distances)
        else:
            return 'NA'
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
    def computeScore(self, dist, num_input_scars):        
        scores = []
        num_lookup_scars = len(list(set([el[0] for el in dist]))) 
        while len(scores) <= num_input_scars - 1:
            if len(scores) >= num_lookup_scars:
                break
            current_lowest = dist[np.argmin([el[1] for el in dist])]        
            if len(dist) != 0:
                scores.append(current_lowest)
            dist = [item for item in dist if item[0] != current_lowest[0]]    
        return np.sum([el[1] for el in scores])


# initiate class:
find_matches_func = Compare_ROIS(path_to_images, None, None, path_to_mask)
find_matches_func.preLoadData()


######################################################################################################


# LITERA
app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])




# dash canvas info
filename = Image.open(path_to_blank)
canvas_width = 259   # this has to be set to 259 because we use the dash as input to the model



score_html = "NA"
n_html = "NA"
num_matches_html = "NA"



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
                                #dbc.CardHeader(html.H6("Toolbox", className="card-text"), style={'textAlign': 'center'}),                                
                                    #html.H6(children=['Orientation'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                
                                    daq.BooleanSwitch(
                                            id='my-boolean-switch',
                                            label="Orientation",
                                            on=True
                                        ),
                                    #html.H6(children=['Width'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                    
                                    daq.BooleanSwitch(
                                            id='my-boolean-switch1',
                                            label="Width",                                            
                                            on=True
                                        ),
                                    #html.H6(children=['Height'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                    
                                    daq.BooleanSwitch(
                                            id='my-boolean-switch2',
                                            label="Height",                                            
                                            on=True
                                        ),                                
                                html.Div([
                                    html.H6(children=['Brush Width'], style={'textAlign': 'center', 'font-weight': 'normal'}),
                                    dcc.Slider(
                                        id='bg-width-slider',
                                        min=1,
                                        max=40,
                                        step=1,
                                        value=1,
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
                                        ),
                                    # dcc.Checklist(
                                    #     options=[
                                    #         {'label': 'New York City', 'value': 'NYC'},
                                    #         {'label': 'Montréal', 'value': 'MTL'},
                                    #         {'label': 'San Francisco', 'value': 'SF'}
                                    #     ],
                                    #     value=['NYC', 'MTL'],
                                    #         labelStyle={'display': 'inline-block', 'margin-right': '20px', 'margin-left': '20px', 'font-weight': 300},
                                    #         inputStyle={"margin-right": "10px"},
                                    #         style={'textAlign': 'left', 'font-weight': 'normal', 'font-size' : '15'}
                                    # ) 
                        ], style = {"margin-top": "10px"}, className="three columns")], style = {'width': '20rem', 'margin-top': '50px', 'margin-right': '100px', 'display': 'inline-block',
                                          'box-shadow': '8px 8px 8px grey',
                                          'margin-bottom': '10px',
                                          'margin-left': '10px'}),    
                        ]), width = 4)]),
                        
                        
                    ], style = {'align-items': 'center', 'width': '50rem', 'margin-left': '40px', 'display': 'inline-block',
                                          'box-shadow': '8px 8px 8px grey',
                                          'margin-bottom': '10px'}), width = 6),               
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
                                                dbc.CardFooter(dbc.Row([
                                                    dbc.Col(html.Div(id = 'sketch_output_info1'), width = 6),
                                                                        dbc.Col(html.Div(id = 'sketch_output_info2'), width = 6)]))
                                ], style={"width": "50rem",
                                          "align-items": "center",
                                          'display': 'inline-block',
                                          'box-shadow': '8px 8px 8px grey',
                                          'margin-bottom': '10px',
                                          'margin-left': '10px'}),  width = 6),
                            ]
                        ),
                    ], style={'padding': '0px 40px 40px 40px'}
                )


last_right = False
blank = base64.b64encode(open(path_to_blank, 'rb').read())

def return_image(n, left = False, init = False):
    global path_to_images
    global names
    global num_returned
    global count
    global last_right
    global score_html, n_html, num_matches_html
        

    if left == False and init == True:
        file = names[n]
        count += 1
        last_right = True

    if left == True and init == True:  
        if last_right == True:
            n = n-2
            last_right = False
        else:
            n = n-1
        count -= 1
        file = names[n]

    
    if init == False:
        file = names[n]
        
    encoded_image = base64.b64encode(open(path_to_images + file, 'rb').read())
    if name_info is None:
        return html.Div([
        html.H5(names[n][0:-4]),
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
        ], style = {'align-items': 'center'})
    else:
        score_html = str(round(name_info[n], 3))
        n_html = str(1+n)
        num_matches_html = str(num_returned)
        return html.Div([
            html.H5(str(file)[0:-4]),
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
            ], style = {'align-items': 'center'})





count = 0
switch = False

@app.callback(
    [Output("sketch_output", "children"),
     Output("sketch_output_info1", "children"),
     Output("sketch_output_info2", "children")],
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
    global score_html, n_html, num_matches_html
    
    if switch == True:
        return_image(0)        
        count = 0
        switch = False   
                    
    if abs(count) == len(names):
        count = 0

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if "right_click" in changed_id:
        return return_image(count, left = False, init = True), html.H6('Score:    ' + str(score_html), style = {'width':'47.5rem', 'textAlign': 'left'}), html.H6('Matches:    ' + str(n_html) + '/' + str(num_matches_html), style = {'textAlign': 'right'})
    if "left_click" in changed_id:
        return return_image(count, left = True, init = True), html.H6('Score:    ' + str(score_html), style = {'width':'47.5rem', 'textAlign': 'left'}), html.H6('Matches:    ' + str(n_html) + '/' + str(num_matches_html), style = {'textAlign': 'right'})
    else:
        return html.Img(src='data:image/png;base64,{}'.format(blank.decode())), html.H6('Score:    ' + str(score_html), style = {'width':'47.5rem', 'textAlign': 'left'}), html.H6('Matches:    ' + str(n_html) + '/' + str(num_matches_html), style = {'textAlign': 'right'})
    
    
    
    
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
        
        mask = parse_jsonstring(string, shape=(559, 259))
        mask = (~mask.astype(bool)).astype(int)
        mask[mask == 1] = 255
        mask = mask.astype(np.uint8)


        find_matches_func.input_sketch = mask
        find_matches_func.roi = bounding_box_list
        matches = find_matches_func.compare_rois()
        t = matches
        is_rect = False
        
        
        names = [i[0] for i in matches]
        name_info = [i[1] for i in matches]
        num_returned = len(matches)        
        
    return





if __name__ == '__main__':
    app.run_server()
