import numpy as np

from utils import image_resize
from video_output_utils import scale_values
from config import*

newpath = ResF + '/' + tname

detected_circles_radius, msk, l1, l2, values1, values2, LK, LK_TM, avheight, prv, prvpp, prv_skl, Aw, perc, percor, er, last_mask_centerline = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
flag, dst_pad, dst_old, d, f = 0, 0, 0, 0, 0
temp_count=[0,0,0]
circle=np.array([0, 0, 0])

mapd=4.6
map_w=round(vid_res*3/2)
wdth_apprx=round(vid_res*0.5625)
map_h=round(map_w/mapd)#

vids_dimensions = [map_w, map_h, mapd, wdth_apprx, vid_res]

tmul = np.float32([[1, 0, 0], [0, 1, 0]])

Nsp=[round(wdth_apprx/2) - round(0.208*wdth_apprx),round(wdth_apprx/2) + round(0.208*wdth_apprx)]
Nep=[round(vid_res/2) - round(0.117*vid_res),round(vid_res/2) + round(0.117*vid_res)]
Zerofill=np.zeros(round(0.208*wdth_apprx)+round(0.117*vid_res))

ul, fl, mp =[np.zeros((vids_dimensions[1], vids_dimensions[0], 3)).astype(np.uint8) for _ in range(3)]
RF_r =[np.zeros((vids_dimensions[1], vids_dimensions[0], 3)).astype(np.uint8) for _ in range(1)]

newfrm, newink = [np.zeros((vids_dimensions[1], vids_dimensions[0])).astype(np.uint8) for _ in range(2)]
LK_TMold = np.zeros([2,3])
LK_TMupd = np.zeros([2,3])

Map_dimension = np.zeros((wdth_apprx, round(wdth_apprx * mapd), 3))
final_dim, ratio = image_resize(Map_dimension, width=vids_dimensions[0])

color_thresholds = {
    'red': [
        {'lower': np.array([0, 150, 50]), 'upper': np.array([10, 255, 255])},  # Range 1
        {'lower': np.array([160, 150, 50]), 'upper': np.array([179, 255, 255])}  # Range 2
    ],
    'blue': [
        {'lower': np.array([85, 100, 150]), 'upper': np.array([95, 255, 255])},   # Range 1
        {'lower': np.array([100, 150, 200]), 'upper': np.array([120, 255, 255])}, # Range 2
        {'lower': np.array([120, 100, 150]), 'upper': np.array([130, 255, 255])}  # Range 3
    ],
    'green': [
        {'lower': np.array([35, 40, 40]), 'upper': np.array([85, 255, 255])},      # Range 1
        {'lower': np.array([45, 52, 40]), 'upper': np.array([90, 255, 255])}       # Range 2
    ],
    'black': [
        {'lower': np.array([0, 0, 0]), 'upper': np.array([360, 255, 50])},      # Range 1
    ]
}
scaling_factor = vid_res / 1280

class Thresholds_frames:
    # Define default variables
    und_l = []
    fnl_p = []
    dtct_r = []

    if custom_range == True:
        und_l = Undistrupted_line
        fnl_p = Final_print
        dtct_r = Detection_range
    else:
        # Automatically set the variables when the class is loaded
        if Dye_selected == 'red':
            und_l = [152, 420]
            fnl_p = [933, 1092]
            dtct_r = [112, 125]
            z_dr = scale_values(26, scaling_factor, is_integer=True)

        elif Dye_selected == 'blue':
            und_l = [95, 366]
            fnl_p = [977, 1242]
            dtct_r = [40, 50]
            z_dr = [0]

        elif Dye_selected == 'green':
            und_l = [0, 0]
            fnl_p = [0, 0]
            dtct_r = [0, 0]
        else:
            raise ValueError(f"Invalid color choice: {Dye_selected}")

resolution_mapping = {
    "350": {
        "Fonts": [1, 0.75, 0.5],
        "Thicknesses": [1, 1, 1],
        "Line_thickness": [9, 1]
    },
    "1280": {
        "Fonts": [4, 3, 2],
        "Thicknesses": [4, 2, 2],
        "Line_thickness": [36, 5]
    }
}
tmfl = np.float32([[1, 0, 0], [0, 1, Thresholds_frames.z_dr[0]]])  # tmul[:, :-1], ([0],[13])


class TextStyle:
    font = scale_values([4, 3, 2], scaling_factor, is_integer=False)
    thickness = scale_values([4, 2, 2], scaling_factor, is_integer=True)
    line_thickness = scale_values([1, 5], scaling_factor, is_integer=True)
