import os
import cv2
import numpy as np
import shutil
from dataprep import*

def create_folders(base_path, subfolders):
    if os.path.exists(base_path):
        user_input = input(
            f"The folder '{base_path}' already exists. Do you want to delete it and create new folders? (y/n): ").lower()
        if user_input == 'y':
            shutil.rmtree(base_path)  # Delete the folder and its contents
            print(f"Deleted the existing folder: {base_path}")
        else:
            print("Operation aborted.")
            exit()

    # Create the base folder and subfolders
    os.makedirs(base_path)
    for folder in subfolders:
        os.makedirs(os.path.join(base_path, folder))
    print(f"Created base folder '{base_path}' and subfolders: {', '.join(subfolders)}")

def initialize_video_writer(newpath, tname, sz, fps):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vout = cv2.VideoWriter()
    vout.open(newpath + '/CCV analysis_' + tname + '.mp4', fourcc, fps, (sz[0] * 3, sz[1] * 2), True)
    canvas_all = np.zeros((sz[1] * 2, sz[0] * 3, 3), dtype=np.uint8)
    return vout, canvas_all

def scale_values(values, scaling_factor, is_integer=False):
    # Ensure values is always a list (even if a single value is passed)
    if not isinstance(values, list):
        values = [values]

    # Apply scaling and rounding based on whether integer rounding is needed
    if is_integer:
        return [round(v * scaling_factor) for v in values]  # Round to nearest integer
    else:
        return [round(v * scaling_factor,2) for v in values]  # Leave as float

def pre_render_static_title(textstyle, maps_dimensions, resolution_w):
    stiching=[]
    for i in (['Area accuracy (%)', 0.05, 0.11], ['Centerline accuracy (mm)', 0.05, 0.11]):
        static_text_img = np.zeros((maps_dimensions[1], int(resolution_w*3/2), 3), dtype=np.uint8)  # Adjust size as needed
        cv2.putText(static_text_img, i[0], (round(i[1] * maps_dimensions[0]), round(i[2] * maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, textstyle.font[0], (255, 255, 255), textstyle.thickness[0], cv2.LINE_AA)
        stiching.append(static_text_img)
    return stiching

def pre_render_static_text(cnt, threshold, textstyle, maps_dimensions, resolution_w):
    static_text_img = np.zeros((maps_dimensions[1], int(resolution_w*3/2), 3), dtype=np.uint8)  # Adjust size as needed
    cv2.putText(static_text_img, 'Undistrupted line: ', (round(0.05 * maps_dimensions[0]), round(0.86*maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, textstyle.font[1], (255, 255, 255), textstyle.thickness[1], cv2.LINE_AA)
    if cnt > threshold:
        cv2.putText(static_text_img, 'Final path: ', (round(0.55*maps_dimensions[0]), round(0.1 * maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, textstyle.font[1], (255, 255, 255), textstyle.thickness[1], cv2.LINE_AA)
    return static_text_img

# Overlay static text and draw dynamic parts
def draw_dynamic_texts(dd_sk, static_text_img, dr, drift, font, thickness, maps_dimensions):
    # Overlay static text
    dd_sk = cv2.addWeighted(dd_sk, 1.0, static_text_img, 1.0, 0)

    # Draw dynamic parts
    cv2.putText(dd_sk, f'{dr} mm', (round(0.3 * maps_dimensions[0]), round(0.86*maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, font, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.putText(dd_sk, f'{drift} mm', (round(0.72*maps_dimensions[0]), round(0.1 * maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, font, (0, 0, 255), thickness, cv2.LINE_AA)

    return dd_sk

def draw_dynamic_textsAcc(cnt, threshold, dd, perc, percor, percentage, percentageor, font, thickness, maps_dimensions):
    # Draw dynamic parts
    cv2.putText(dd, f'{perc} %', (round(0.3 * maps_dimensions[0]), round(0.86 * maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, font,(0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(dd, f'{percor} %', (round(0.4 * maps_dimensions[0]), round(0.86 * maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, font,(0, 0, 255), thickness, cv2.LINE_AA)
    if cnt > threshold:
        cv2.putText(dd, f'{percentage} %', (round(0.72 * maps_dimensions[0]), round(0.1 * maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, font, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.putText(dd, f'{percentageor} %', (round(0.8 * maps_dimensions[0]), round(0.1 * maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, font, (0, 0, 255), thickness, cv2.LINE_AA)
    return dd

def video_output(multiwindow_frame, sz, original, frame, ink, dd, dd_sk, resolution_w):
    multiwindow_frame[:sz[1], :sz[0]] = original
    multiwindow_frame[:sz[1], sz[0]:sz[0] * 2] = frame  # dst
    multiwindow_frame[:sz[1], sz[0] * 2:] = ink  # dst#sb#mm
    multiwindow_frame[sz[1]:sz[1] + dd.shape[0], :2 * sz[0] - int(resolution_w/2)] = dd
    multiwindow_frame[sz[1]:sz[1] + dd_sk.shape[0], sz[0]*2 - int(resolution_w/2):] = dd_sk

    return multiwindow_frame

def snaps(cnt, length, original, dd, dd_sk, ink, frame, newpath, tname, tf):
    if cnt == int(tf.und_l[0]/2):
        cv2.imwrite(newpath + '/Nozzle/Frame_' + str(cnt) + '_' + tname + '.jpg', original)
    elif cnt == tf.und_l[1]-1 or cnt == length - 2:
        cv2.imwrite(newpath + '/0_Area accuracy ' + str(cnt) + '_ovl_' + tname + '.jpg', dd)
        cv2.imwrite(newpath + '/1_Centerline drift ' + str(cnt) + '_avr_' + tname + '.jpg', dd_sk)

        # white
        dd[np.where((dd == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        dd_sk[np.where((dd_sk == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        cv2.imwrite(newpath + '/0_Area accuracy ' + str(cnt) + '_ovl_W_' + tname + '.jpg', dd)
        cv2.imwrite(newpath + '/1_Centerline drift ' + str(cnt) + '_avr_W_' + tname + '.jpg', dd_sk)
    elif cnt == tf.und_l[0]+1:
        cv2.imwrite(newpath + '/Steps/0_Original frame ' + str(cnt) + '_' + tname + '.jpg', original)
        cv2.imwrite(newpath + '/Steps/1_Enhanced contrast ' + str(cnt) + '_' + tname + '.jpg', frame)
        cv2.imwrite(newpath + '/Steps/2_Ink' + str(cnt) + '_' + tname + '.jpg', ink)
        cv2.imwrite(newpath + '/Steps/3_Area accuracy ' + str(cnt) + '_' + tname + '.jpg', dd)
        cv2.imwrite(newpath + '/Steps/4_Centerline drift ' + str(cnt) + '_' + tname + '.jpg', dd_sk)
