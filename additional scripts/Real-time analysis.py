#!/usr/bin/python3
import cv2
import math
import numpy as np
import os
import time
import random
import serial
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

os.environ["LIBCAMERA_LOG_LEVELS"] = "3"

ResF = './CV_video_RTv4_7'  # Thickness16'
tname = 't1_05_1400'
file1 = open('w285h285_centered.gcode', 'r')
Ym1="G1 Y-5 E-2"#   
Ym2="G1 Y5 E-2"#   
Ym3="G1 Y1.07"#   Ym3=0.5+0.285*2=1.07       old 0.34+0.7=1.04
Yt="G1 Y-0.285"#  Yt=0.285                   old 0.34/2=0.17

mapd = 4.6
resolution_w = 320  # 1280
map_w = round(resolution_w * 3 / 2)
wdth_apprx = round(resolution_w * 0.5625)
map_h = round(map_w / mapd)  #
tmul = np.float32([[1, 0, 16], [0, 1, -4]])
tmfl = np.float32([[1, 0, 12], [0, 1, 4]])  # tmul[:, :-1], ([0],[13])
# crop nozzle from LK
Nsp = [round(wdth_apprx / 2) - round(0.208 * wdth_apprx), round(wdth_apprx / 2) + round(
    0.208 * wdth_apprx)]  # Nsp=[round(wdth_apprx/2) - round(0.208*wdth_apprx),round(wdth_apprx/2) + round(0.208*wdth_apprx)]
Nep = [round(resolution_w / 2) - round(0.117 * resolution_w), round(resolution_w / 2) + round(
    0.117 * resolution_w)]  # Nep=round(resolution_w/2) - round(0.117*resolution_w),round(resolution_w/2) + round(0.117*resolution_w)
Zerofill = np.zeros(round(0.208 * wdth_apprx) + round(
    0.117 * resolution_w))  # Zerofill=np.zeros(round(0.208*wdth_apprx)+round(0.117*resolution_w))

CDlineTh = 1
IAlineTh = 1

ref_wdth = 0.428  # 428#0.285
fontSc = 1
thickn = 1
LK = []
avheight, prvpp = [], []
SourceF = './Prints'
if not os.path.exists(ResF):
    os.makedirs(ResF)
else:
    print('Folder already exists. Delete to run again..')
    exit()

File_name = os.listdir('./' + SourceF)
time.sleep(1)

picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

encoder = H264Encoder(10000000)

time.sleep(1)


def gcode(code):
    ser = serial.Serial("/home/pi/printer_data/comms/klippy.serial", 250000)
    tx = f'{code} \n'
    ser.write(bytearray(tx.encode('utf-8')))
    ser.flush()


def pre_render_static_text(fontSc, thickn):
    static_text_img = np.zeros((map_h, int(resolution_w * 3 / 2), 3), dtype=np.uint8)  # Adjust size as needed
    cv2.putText(static_text_img, 'Undistrupted line: ', (round(0.05 * map_w), round(0.86 * map_h)),
                cv2.FONT_HERSHEY_PLAIN, fontSc, (255, 255, 255), thickn, cv2.LINE_AA)
    cv2.putText(static_text_img, 'Final path: ', (round(0.58 * map_w), round(0.15 * map_h)), cv2.FONT_HERSHEY_PLAIN,
                fontSc, (255, 255, 255), thickn, cv2.LINE_AA)
    return static_text_img


# Overlay static text and draw dynamic parts
def draw_dynamic_texts(dd_sk, static_text_img, dr, drift, fontSc, thickn):
    # Overlay static text
    dd_sk = cv2.addWeighted(dd_sk, 1.0, static_text_img, 1.0, 0)

    # Draw dynamic parts
    cv2.putText(dd_sk, f'{dr} mm', (round(0.38 * map_w), round(0.86 * map_h)), cv2.FONT_HERSHEY_PLAIN, fontSc,
                (0, 255, 0), thickn, cv2.LINE_AA)
    cv2.putText(dd_sk, f'{drift} mm', (round(0.78 * map_w), round(0.15 * map_h)), cv2.FONT_HERSHEY_PLAIN, fontSc,
                (0, 255, 0), thickn, cv2.LINE_AA)

    return dd_sk


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size
    (h, w) = image.shape[:2]

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # Calculate the aspect ratio
    aspect_ratio = float(w) / h

    # Calculate new dimensions based on width or height
    if width is None:
        new_width = round(height * aspect_ratio)
        new_height = height
    elif height is None:
        new_width = width
        new_height = round(width / aspect_ratio)
    else:
        new_width = width
        new_height = height

    # Resize the image
    resized = cv2.resize(image, (new_width, new_height), interpolation=inter)

    return resized


def dscale(image, Blower, Bupper):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the image to obtain a mask
    mask = cv2.inRange(hsv, Blower, Bupper)

    # Apply median blur to the mask
    mask_blurred = cv2.medianBlur(mask, 9)

    # Detect circles using Hough transform
    # circles = cv2.HoughCircles(mask_blurred, cv2.HOUGH_GRADIENT, 8, 1, param1=100, param2=50)
    #circles = cv2.HoughCircles(mask_blurred, cv2.HOUGH_GRADIENT, 8, 1, param1=70, param2=30)
    circles = cv2.HoughCircles(mask_blurred, cv2.HOUGH_GRADIENT, 8, 1, param1=130, param2=60) # Best for 320? maybe


    # If a circle is found, convert its parameters to integers and draw it on the image
    if circles is not None:
        circle = circles[0, 0]
        x, y, r = map(int, circle)
        cv2.circle(image, (x, y), r, (0, 255, 0), thickn*3)

        # Convert circle parameters to integers
        circles = circle.astype(int)

    return circles, image, mask_blurred


def Fwidth(image, circle, cnt, newpath):
    x, y, r = circle[:]  # Simplify tuple unpacking
    # Extract the region of interest from the image
    img = image[y - round(0.21 * wdth_apprx):y + round(0.21 * wdth_apprx),
          x - round(0.2578 * resolution_w):x - round(0.0625 * resolution_w)]
    # Optionally, save the extracted region as an image
    if cnt == th:
        cv2.imwrite(f"{newpath}/Mask_example_{tname}.jpg", img)
    # Count the non-zero pixels along the columns
    how_many_c = np.count_nonzero(img, axis=0)
    # Calculate the average width
    real_hmc = how_many_c[how_many_c != 0]
    avwidthc = np.average(real_hmc / (2 * r)) * ref_wdth
    return round(avwidthc, 3)  # Round only when returning the result


def refpath(Limg, cnt, prvpp, color=(255, 0, 0), thickness=2):
    # print(cnt)
    positions = np.nonzero(Limg.T)
    if cnt < th:
        if not positions[1].any():
            ref = np.zeros_like(Limg)
        else:
            avheight.append(round((min(positions[2]) + max(positions[2])) / 2))
            p1 = (min(positions[1]), round(np.average(avheight) * 1.03))
            p2 = (max(positions[1]), round(np.average(avheight) * 1.03))
            scl = (p2[0] - p1[0]) / 18
            ref = cv2.line(np.zeros_like(Limg), p1, p2, color, thickness)
            prvpp = [p1, p2, scl]
    else:
        p3 = tuple(np.subtract(prvpp[1], (0, round(1.7 * prvpp[2]))))
        p4 = tuple(np.subtract(p3, (round(3 * prvpp[2]), 0)))
        p5 = tuple(map(sum, zip(p4, (0, round(0.5 * prvpp[2])))))
        p6 = tuple(np.subtract(p5, (round(3 * prvpp[2]), 0)))
        p7 = tuple(map(sum, zip(p6, (0, round(0.5 * prvpp[2])))))
        p8 = tuple(np.subtract(p7, (round(3 * prvpp[2]), 0)))

        ref = cv2.line(np.zeros_like(Limg), prvpp[0], prvpp[1], color, thickness)
        ref = cv2.line(ref, prvpp[1], p3, color, thickness)
        # ref = cv2.addWeighted(ref, alpha, msk, 1 - alpha, 0)
        ref = cv2.line(ref, p3, p4, color, thickness)
        # ref = cv2.addWeighted(ref, alpha, msk, 1 - alpha, 0)
        ref = cv2.line(ref, p4, p5, color, thickness)
        # ref = cv2.addWeighted(ref, alpha, msk, 1 - alpha, 0)
        ref = cv2.line(ref, p5, p6, color, thickness)  # -168
        # ref = cv2.addWeighted(ref, alpha, msk, 1 - alpha, 0)
        ref = cv2.line(ref, p6, p7, color, thickness)  # -28
        # ref = cv2.addWeighted(ref, alpha, msk, 1 - alpha, 0)
        ref = cv2.line(ref, p7, p8, color, thickness)  # -168
        # ref = cv2.addWeighted(ref, alpha, msk, 1 - alpha, 0)

    return ref, prvpp


def crtmap2(comparison, segment):
    merge_skel = []
    cmp = comparison
    # for i, cmp in enumerate(comparison):
    if len(cmp.shape) > 2:
        img = cv2.cvtColor(cmp, cv2.COLOR_BGR2GRAY)
    else:
        img = cmp
    _, gray = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(gray, (5, 5), 0)
    line = cv2.ximgproc.thinning(img)
    if segment == 0:
        line = cv2.merge((np.zeros_like(line), np.zeros_like(line), line))
    elif segment == 1:
        # img=255-img
        line = cv2.merge((np.zeros_like(line), line, np.zeros_like(line)))

    # merge_skel.append(mdskl)
    return line  # merge_skel


def averages(p):
    unique_positions, indices = np.unique(p[0], return_inverse=True)
    sums = np.zeros_like(unique_positions, dtype=np.float64)
    counts = np.zeros_like(unique_positions, dtype=np.int32)
    np.add.at(sums, indices, p[1])
    np.add.at(counts, indices, 1)
    averages = sums / counts
    return np.column_stack((unique_positions, averages)).reshape(-1, 2)


def Ferror(reference, image, circle, ref_wdth):
    drift = []

    # Segmentation
    hl1_r, hl1_i = reference[round(0.3589 * map_h):, :round(0.91 * map_w)], image[round(0.3589 * map_h):,
                                                                            :round(0.91 * map_w)]
    hl234_r, hl234_i = reference[:round(0.3589 * map_h), :round(0.91 * map_w)], image[:round(0.3589 * map_h),
                                                                                :round(0.91 * map_w)]
    vl1_r, vl1_i = reference[:, round(0.91 * map_w):], image[:, round(0.91 * map_w):]
    Segm = [(hl1_r, hl1_i), (hl234_r, hl234_i), (vl1_r, vl1_i)]
    for i, (seg_r, seg_i) in enumerate(Segm):
        # Non-zero positions
        positions_ref = np.nonzero(seg_r.T) if i < 2 else np.nonzero(seg_r)
        positions_img = np.nonzero(seg_i.T) if i < 2 else np.nonzero(seg_i)

        if not positions_img[0].any() or not positions_ref[0].any():
            #time.sleep(avlpt)
            continue

        # Calculate averages
        res_r = averages(positions_ref)
        res_i = averages(positions_img)

        # Get the minimum size
        ms = min(res_i.shape[0], res_r.shape[0])

        if res_i[0, 0] > res_r[0, 0]:
            startP = np.searchsorted(res_r[:, 0], res_i[0, 0])
            endP = min(ms + startP, res_r.shape[0])
            if endP > res_i.shape[0]:
                drift_segment = np.mean(res_r[startP:startP+res_i.shape[0], 1] - res_i[:ms, 1])
            else:
                # print(cnt)
                # if cnt>=1042:
                #    print('opa')
                drift_segment = np.mean(res_r[startP:endP, 1] - res_i[:ms - startP, 1])
        else:
            startP = np.searchsorted(res_i[:, 0], res_r[0, 0])
            endP = min(ms + startP, res_i.shape[0])
            if endP > res_i.shape[0] or startP == endP:
                #time.sleep(avlpt)
                continue
            else:
                drift_segment = np.mean(res_i[startP:endP, 1] - res_r[:endP - startP, 1])

        drift.append(drift_segment)

    r = circle[2]
    # print("Drift values:", drift)
    if not drift:
        error = None
    else:
        drift = [x for x in drift if str(x) != 'nan']
        error = round((sum(map(abs, drift)) / len(drift) / (2 * r)) * ref_wdth, 3)
    # print("Error:", error)
    return error


def mapping(oldfrm, newfrm, cnt):
    # newfrm[0:4, 0:] = 0
    # oldfrm[0:4, 0:] = 0
    newfrm = newfrm[:, round(0.1 * resolution_w):round(1 * resolution_w / 2)]
    oldfrm = oldfrm[:, round(0.1 * resolution_w):round(1 * resolution_w / 2)]
    listX, listY = [], []
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=500, qualityLevel=0.85, minDistance=np.round(0.02 * resolution_w).astype(int),
                          blockSize=100)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(np.round(0.17 * resolution_w).astype(int), np.round(0.17 * resolution_w).astype(int)),
                     maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Random colors propably for the lines
    color = np.random.randint(0, 255, (1000, 3))

    if len(oldfrm.shape) > 2:
        GRold = cv2.cvtColor(oldfrm, cv2.COLOR_BGR2GRAY)
    else:
        GRold = oldfrm
    p0 = cv2.goodFeaturesToTrack(GRold, mask=None, **feature_params)

    mask = np.zeros_like(GRold)
    mask = cv2.merge((mask, mask, mask))
    frame = cv2.merge((newfrm, newfrm, newfrm))
    if len(newfrm.shape) > 2:
        GRnew = cv2.cvtColor(newfrm, cv2.COLOR_BGR2GRAY)
    else:
        GRnew = newfrm
    if p0 is None:
        p1=None
    else:
        p1, st, err = cv2.calcOpticalFlowPyrLK(GRold, GRnew, p0, None, **lk_params)
    if p1 is not None:
        Gnew = p1[st == 1]
        Gold = p0[st == 1]

        for i, (new, old) in enumerate(zip(Gnew, Gold)):
            a, b = new.ravel()
            c, d = old.ravel()

            listX.append(a-c)
            listY.append(b-d)
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)
        list_X= np.array(listX)
        listX=reject_outliers(list_X)
        list_Y= np.array(listY)
        listY=reject_outliers(list_Y)
    if len(listX)!=0 or listX is None:
        LK_X=sum(listX) / len(listX)
        LK_Y=sum(listY) / len(listY)
    else:
        LK_X, LK_Y = 0, 0
        img=np.zeros_like(frame)

    cv2.imwrite(newpath + '/LK/LK ' + str(cnt) + '.jpg', img)
    TM = np.float32([[1, 0, int(LK_X)], [0, 1, int(LK_Y)]])
    return TM


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else np.zeros(len(d))
    return data[s < m]


def warpPerspectivePadded(src, dst, transf):
    src_h, src_w = src.shape[:2]
    lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])

    trans_lin_homg_pts = transf.dot(lin_homg_pts)

    minX = np.min(trans_lin_homg_pts[0, :])
    minY = np.min(trans_lin_homg_pts[1, :])
    maxX = np.max(trans_lin_homg_pts[0, :])
    maxY = np.max(trans_lin_homg_pts[1, :])

    dst_h, dst_w = dst.shape[:2]
    pad_h = int(np.round(np.maximum(dst_h, maxY) - np.minimum(0, minY)))
    pad_w = int(np.round(np.maximum(dst_w, maxX) - np.minimum(0, minX)))

    pad_sz = (pad_h, pad_w) + dst.shape[2:]
    dst_pad = np.zeros(pad_sz, dtype=np.uint8)

    anchorX = int(np.round(-minX)) if minX < 0 else 0
    anchorY = int(np.round(-minY)) if minY < 0 else 0

    transl_transf = np.eye(3)
    transl_transf[0, 2] += anchorX
    transl_transf[1, 2] += anchorY

    dst_pad[anchorY:anchorY + dst_h, anchorX:anchorX + dst_w] = dst

    return dst_pad


def detectnozzletip(resolution_w, wdth_apprx, min_percent, max_percent, cnt, blower, bupper):
    for i in range(100):
        cnt += 1

        data8 = picam2.capture_array('main')
        frame = data8[:, :, [2, 1, 0]]
        frame = image_resize(frame, resolution_w)
        original = frame.copy()
        cv2.putText(original, f'Frame number: {cnt}', (round(0.1 * resolution_w), round(0.2 * wdth_apprx)),
                    cv2.FONT_HERSHEY_PLAIN, fontSc, (0, 255, 0), thickn, cv2.LINE_AA)

        lo, hi = np.percentile(frame, (min_percent, max_percent))
        # Apply linear "stretch" - lo goes to 0, and hi goes to 1
        res_img = (frame.astype(float) - lo) / (hi - lo)
        # Multiply by 255, clamp range to [0, 255] and convert to uint8
        frame = np.maximum(np.minimum(res_img * 255, 255), 0).astype(np.uint8)

        cir, im, mk = dscale(frame, blower, bupper)
        if cir is not None:
            circle = cir
            cirr.append(cir[2])
            circle[2] = np.average(cirr)
            mk = cv2.merge((mk, mk, mk))
            HTres = np.concatenate((mk, im), axis=1)
            cv2.imwrite(newpath + '/Nozzle/Nozzle detection example ' + str(cnt) + '_msk.jpg', HTres)
            cv2.imshow('Detected Nozzle Tip', HTres)
            cv2.waitKey(1)
        #else:
        #    cv2.imshow('No detection', frame)
        #    cv2.waitKey(1)            

    # closing all open windows
    cv2.destroyAllWindows()
    return circle


Rlower1 = np.array([0, 150, 50])
Rupper1 = np.array([10, 255, 255])

Rlower2 = np.array([160, 150, 50])
Rupper2 = np.array([179, 255, 255])

blower = np.array([0, 0, 0])
bupper = np.array([350, 66, 90])

min_percent = 1  # Low percentile
max_percent = 95  # High percentile

ul, fl, mp = [np.zeros((map_h, map_w, 3)).astype(np.uint8) for _ in range(3)]
RF_r = [np.zeros((map_h, map_w, 3)).astype(np.uint8) for _ in range(1)]

avlpt=0
looptime=[]
# RFa = refpath(np.zeros(mp.shape), 1, thickness=1)  # .astype(np.uint8)
# mask_RFa = RFa != [0, 0, 0]
# RFb = refpath(np.zeros(mp.shape), 500, thickness=1)  # .astype(np.uint8)
# mask_RFb = RFb != [0, 0, 0]
# RF_ra = refpath(np.zeros(mp.shape), 1, color=(255, 0, 0), thickness=18).astype(np.uint8)
# mask_RF_ra = RF_ra != [0, 0, 0]
# RF_rb = refpath(np.zeros(mp.shape), 500, color=(255, 0, 0), thickness=28).astype(np.uint8)
# mask_RF_rb = RF_rb != [0, 0, 0]

static_text_img = pre_render_static_text(fontSc, thickn)
############################################################################################################################
cnt = 0  # 1150#1180
flag = 0

newpath = ResF + '/' + tname
if not os.path.exists(newpath):
    os.makedirs(newpath)
    os.makedirs(newpath + '/Nozzle')
    os.makedirs(newpath + '/Steps')
    os.makedirs(newpath + '/Mapping')
    os.makedirs(newpath + '/Mapping/Final')
    os.makedirs(newpath + '/Mapping/Undstr')
    os.makedirs(newpath + '/LK')

Aw, L3, L4, L5 = [], [], [], []

output = FfmpegOutput(newpath + '/' + tname + '.mp4')
picam2.start_recording(encoder, output)
time.sleep(1)

data8 = picam2.capture_array('main')

frame = data8[:, :, [2, 1, 0]]
frame = image_resize(frame, resolution_w)
## some videowriter props
sz = frame.shape

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

## open and set props
vout = cv2.VideoWriter()
vout.open(newpath + '/CCV analysis_' + tname + '.mp4', fourcc, 6, (sz[1] * 3, sz[0] + map_h), True)
canvas_all = np.zeros((sz[0] + map_h, sz[1] * 3, 3), dtype=np.uint8)
# alpha = 0.8
point_1 = int(sz[0] * 0.56)
point_2 = int(sz[0] * 0.7)

# length=630
# cv2.imwrite('First frame ' + str(tname) + '.jpg', frame)
# circle = np.array([0])

newfrm, newink = [np.zeros((map_h, map_w)).astype(np.uint8) for _ in range(2)]
dst_pad = np.zeros((map_h, int(sz[0] * 3 / 2), 3), dtype=np.uint8)
nr, nc = 720, 3300
LK_TMold = np.zeros([2, 3])
LK_TMupd = np.zeros([2, 3])

d, f = 0, 0

gcode("G90")
gcode("G1 Z25 F1000")
input("It is about to Home All Axes? Should it continue? Press Enter...")
input("Sure? Press Enter...")

gcode("G28 W")
gcode("BED_MESH_PROFILE LOAD=060224")

input("Press Enter to Go at Center...")
gcode("G1 Z25 F1000")
gcode("G1 X68 Y65 F1000")
input("Press Enter to Run Pre-print and bring Nozzle Tip to Start Position...")

z_o = 3
# file1 = open('w285h285.gcode', 'r')
Lines = file1.readlines()

flaggc = 0
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    if line.strip() == 'G4 ; wait':
        break
    if line[0]==";" or line[0]==" " or line[0]=="\n" or line[0:3]=="EXC" or line[0:3]=="M73" or line[0:4]=="M204" or line[0:4]=="M107" or line[0:4]=="M600" or not line:
        continue
    if line[0:4] == "G1 X" and flaggc == 0:
        flaggc = 1
        # gcode("G1 Z25 F1200")
        # print("G1 Z25 F1200")
    if line[0:4] == "G1 Z":
        com = line.split()
        tt = str.maketrans('', '',
                           'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~')
        ns = com[1].translate(tt)
        Offset_Z = float(ns) + z_o

        com[1] = 'Z' + str(Offset_Z)
        new_line = ' '.join(com)
        gcode(new_line)
        print(new_line)
        continue
    gcode(line)
    print("Line{}: {}".format(count, line.strip()))
    if flaggc == 1:
        flaggc = 2

        gcode("G90")
        gcode("G1 Z7 F100")
        input("Do you like the height? Press Enter...")
        gcode("G91")
        gcode("G1 X10 Y10 F100")
        gcode("G1 X-5 E8 F100")
        gcode("G1 Y-2 F100")
        gcode("G1 X-5 F100")
        gcode("G1 Y-2 F100")
        gcode("G1 X5 F100")
        gcode("G1 Y-2 F100")
        gcode("G1 X-5 F100")
        gcode("G1 Y-4 F100")
        input("Ready for the dive? Press Enter...")
        gcode("G90")
        gcode("G1 Z3.285 F100")
        input("Nozzle detection! Ready? Press Enter...")
        cnt = 0
        cir = np.array([0])
        cirr = []
        circle = detectnozzletip(resolution_w, wdth_apprx, min_percent, max_percent, cnt, blower, bupper)
        print(circle)
        input("Press Enter to Run Gcode and CCV Analysis...")

file1.close()

gcode("G91")
gcode("G1 E-1 F400")
gcode("G1 X-2 F400")
gcode(Ym1)
gcode("G1 X-1")
gcode(Ym2)
gcode(Ym3)
gcode("G1 X-3")

gcode("G1 Z15 F150")
gcode(Yt)
gcode("G1 X15")
gcode("G1 Y-1.7")
gcode("G1 X-3")
gcode("G1 Y0.5")
gcode("G1 X-3")
gcode("G1 Y0.5")
gcode("G1 X-6")
gcode("G1 Y0.7")
gcode("G1 X-3")

print("File finished.")


cnt = 0
thmp = 13
th = 67#53#68#86
thfl=140#119#140#175
thflend=162#143#162#202
while cnt < 260:#205
    start_t = time.time()
    cnt += 1

    data8 = picam2.capture_array('main')
    frame = data8[:, :, [2, 1, 0]]
    frame = image_resize(frame, resolution_w)
    original = frame.copy()
    cv2.putText(original, f'Frame {cnt}', (round(0.01 * resolution_w), round(0.2 * wdth_apprx)), cv2.FONT_HERSHEY_PLAIN,
                fontSc, (0, 255, 0), thickn, cv2.LINE_AA)

    lo, hi = np.percentile(frame, (min_percent, max_percent))
    # Apply linear "stretch" - lo goes to 0, and hi goes to 1
    res_img = (frame.astype(float) - lo) / (hi - lo)
    # Multiply by 255, clamp range to [0, 255] and convert to uint8
    frame = np.maximum(np.minimum(res_img * 255, 255), 0).astype(np.uint8)
    if cnt == 1:
        IAlineTh = np.round(((2 * circle[2]) / ref_wdth * 0.285) / (wdth_apprx / map_h)).astype(int)

    ###############################################################################################
    ################# ~ Width ~ #########################
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_mask = cv2.inRange(hsv, Rlower1, Rupper1)
    upper_mask = cv2.inRange(hsv, Rlower2, Rupper2)
    mask = lower_mask + upper_mask

    imask = mask > 0
    ink = np.zeros_like(frame, np.uint8)
    ink[imask] = frame[imask]
    ink_tmp = ink.copy()

    if circle.shape[0] > 2:
        if cnt <= th:
            avwidthc = Fwidth(mask, circle, cnt, newpath)
            cv2.putText(ink, f'Width: {avwidthc} mm', (round(0.05 * resolution_w), round(0.15 * wdth_apprx)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontSc, (0, 255, 0), thickn, cv2.LINE_AA)

            if not math.isnan(avwidthc) and avwidthc < 2:
                Aw.append(avwidthc)

        else:
            cv2.putText(ink, f'Av. Width:{round(np.average(Aw), 3)} mm',
                        (round(0.05 * resolution_w), round(0.15 * wdth_apprx)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontSc, (0, 255, 0), thickn, cv2.LINE_AA)

    #############################################################################################################################################
    ################# ~ Full print ~ #########################

    #############################################################################################################################################
    ###### ~ Compare ~ ########
    oldfrm = newfrm
    oldink = newink
    newfrm = crtmap2(ink_tmp, 2)
    newink = ink_tmp.copy()
    newfrm[Nsp[0]:Nsp[1], Nep[0]:Nep[1]] = Zerofill
    if oldfrm is not None and (cnt > thmp or cnt > thfl):
        if cnt >= thfl and flag == 0:
            dst_old = np.zeros_like(oldink)
            dst_pad = np.zeros_like(oldink)
            LK_TMupd = np.concatenate((LK_TM[:, :-1], np.zeros([2, 1])), axis=1)
            LK_TMold = np.concatenate((LK_TM[:, :-1], np.zeros([2, 1])), axis=1)
            flag = 1
        if cnt < th or (cnt >= thfl and cnt < thflend):
            #if cnt < th:
            LK_TM = mapping(oldfrm, newfrm, cnt)
            LK.append(LK_TM)
            #if cnt > th:
            #    LM = random.choice(LK)  # np.round(sum(LK) / len(LK))
            #    LK_TM = np.round(np.concatenate((LM[:, :-1], LM[:, -1].reshape(2, 1) * 1.65), axis=1))

            tm = LK_TM[:, -1] + LK_TMold[:, -1]
            tm = tm.reshape(2, 1)
            LK_TMupd = np.concatenate((LK_TM[:, :-1], tm), axis=1)

            if abs(LK_TMupd[0][2]) <= abs(LK_TMold[0][2]):
                #cnt=cnt-1
                #time.sleep(avlpt)
                time.sleep(0.1)
                continue

            dst_pad = warpPerspectivePadded(oldink, newink, LK_TMupd)
            h = np.min([dst_pad.shape[0], oldink.shape[0]])

            if (LK_TM == LK_TMupd).all():
                dst_pad[:h, :abs(int(LK_TMupd[0][2]))] = oldink[:h, :abs(int(LK_TM[0][2]))]
            else:
                if int(LK_TMupd[1][2]) == int(LK_TMold[1][2]):
                    # if dst_pad[:h,  abs(int(LK_TMold[0][2])):abs(int(LK_TMupd[0][2]))].shape == oldink[:h, :abs(int(LK_TM[0][2]))].shape:
                    dst_pad[:h, abs(int(LK_TMold[0][2])):abs(int(LK_TMupd[0][2]))] = oldink[:h, :abs(int(LK_TM[0][2]))]
                    dst_pad[:h, :abs(int(LK_TMold[0][2]))] = dst_old[:h, :abs(int(LK_TMold[0][2]))]
                else:
                    # if dst_pad[:h,  abs(int(LK_TMold[0][2])):abs(int(LK_TMupd[0][2]))].shape == oldink[:h, :abs(int(LK_TM[0][2]))].shape:
                    dst_pad[:h, abs(int(LK_TMold[0][2])):abs(int(LK_TMupd[0][2]))] = oldink[:h, :abs(int(LK_TM[0][2]))]
                    dst_pad[:h, :abs(int(LK_TMold[0][2]))] = dst_old[:h, :abs(int(LK_TMold[0][2]))]
            if cnt < th:
                UL = cv2.warpAffine(dst_pad, tmul, (round(wdth_apprx * mapd), wdth_apprx), borderValue=(0, 0, 0))
                ul = image_resize(UL, width=map_w)
                cv2.imwrite(newpath + '/Mapping/Undstr/Mapping progress' + str(cnt) + '.jpg', ul)
            elif cnt > thfl and cnt < thflend:
                FL = cv2.warpAffine(dst_pad, tmfl, (round(wdth_apprx * mapd), wdth_apprx), borderValue=(0, 0, 0))
                fl = image_resize(FL, width=map_w)
                cv2.imwrite(newpath + '/Mapping/Final/Mapping progress' + str(cnt) + '.jpg', fl)
            tm = LK_TM[:, -1] + LK_TMold[:, -1]
            tm = tm.reshape(2, 1)
            LK_TMold = np.concatenate((LK_TM[:, :-1], tm), axis=1)
            dst_old = dst_pad.copy()

    if cnt < th:
        mp = ul  # merge#[0]#cv2.addWeighted(merge[0], alpha, merge[1], 1 - alpha, 0)
        msk = crtmap2(ul, 0)  # merge_skel_#[0].copy()
        prv = mp  # merge#[0].copy()
        prv_skl = msk  # merge_skel_#[0].copy()
        mk = msk.copy()
    else:
        mp = fl  # merge#[1]
        msk = crtmap2(fl, 1)  # merge_skel_#[1].copy()
        mk = msk.copy()

    if cnt >= th:
        prv_ink = prv_skl != [0, 0, 0]
        msk[prv_ink] = prv_skl[prv_ink]

    mask_ink = mp != [0, 0, 0]
    # Color the pixels in the mask
    if cnt < th:
        RFa, prvpp = refpath(msk.copy(), cnt, prvpp, thickness=CDlineTh)  # .astype(np.uint8)
        mask_RFa = RFa != [0, 0, 0]
        RF_ra, prvpp = refpath(mp.copy(), cnt, prvpp, color=(255, 0, 0), thickness=IAlineTh)
        mask_RF_ra = RF_ra != [0, 0, 0]

        RF_rc = RF_ra.copy()
        msk[mask_RFa] = RFa.copy()[mask_RFa]
        RF_rc[mask_ink] = mp[mask_ink]
        RF_grey = cv2.cvtColor(RFa.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        RF = RF_ra.copy()
    else:
        RFb, prvpp = refpath(msk.copy(), cnt, prvpp, thickness=CDlineTh)  # .astype(np.uint8)
        mask_RFb = RFb != [0, 0, 0]
        RF_rb, prvpp = refpath(mp.copy(), cnt, prvpp, color=(255, 0, 0), thickness=IAlineTh)
        mask_RF_rb = RF_rb != [0, 0, 0]

        RF_rc = RF_rb.copy()
        msk[mask_RFb] = RFb.copy()[mask_RFb]
        RF_rc[mask_ink] = mp[mask_ink]
        RF_grey = cv2.cvtColor(RFb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        RF = RF_rb.copy()

    if cnt >= th:
        RF_rc = cv2.addWeighted(prv, 0.6, RF_rc, 1 - 0.6, 0)
        msk = cv2.addWeighted(prv_skl, 0.2, msk, 1 - 0.2, 0)

    dd = RF_rc
    dd_sk = msk

    cv2.putText(dd_sk, 'Centerline accuracy (mm)', (round(0.05 * map_w), round(0.15 * map_h)), cv2.FONT_HERSHEY_PLAIN,
                fontSc,
                (255, 255, 255), thickn, cv2.LINE_AA)
    cv2.putText(dd, 'Area accuracy (%)', (round(0.05 * map_w), round(0.15 * map_h)), cv2.FONT_HERSHEY_PLAIN,
                fontSc, (255, 255, 255), thickn, cv2.LINE_AA)
    if cnt > thmp:
        Q_grey = cv2.cvtColor(mk, cv2.COLOR_RGB2GRAY)
        bitwiseand = cv2.bitwise_and(RF_grey, Q_grey)
        _, bitwiseand = cv2.threshold(bitwiseand, 0, 255, cv2.THRESH_BINARY)

        Q_grey = cv2.cvtColor(mp.copy(), cv2.COLOR_RGB2GRAY)
        RF_r_grey = cv2.cvtColor(RF.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        _, Q_grey = cv2.threshold(Q_grey, 0, 255, cv2.THRESH_BINARY)
        bitwiseand = cv2.bitwise_and(RF_r_grey, Q_grey)
        bitwiseor = cv2.bitwise_or(RF_r_grey, Q_grey)
        _, bitwiseand = cv2.threshold(bitwiseand, 0, 255, cv2.THRESH_BINARY)
        _, RF_r_bin = cv2.threshold(RF_r_grey, 0, 255, cv2.THRESH_BINARY)
        _, bitwiseor = cv2.threshold(bitwiseor, 0, 255, cv2.THRESH_BINARY)
        bitwiseor = bitwiseor - RF_r_bin

        total = RF_r_grey  # + Q_grey
        total_pixels = total[total > 0].shape[0]
        if total_pixels > 0:
            matches = bitwiseand[bitwiseand > 0].shape[0]
            matchesor = bitwiseor[bitwiseor > 0].shape[0]
            percentage = round((100 * matches / total_pixels), 2)
            percentageor = round((100 * matchesor / total_pixels), 2)

        else:
            percentage = 0
            percentageor = 0

        if cnt < th:
            ULerror = Ferror(RF_grey, Q_grey, circle, ref_wdth)

            er = ULerror.copy()
            dd_sk = cv2.addWeighted(dd_sk, 1.0, static_text_img, 1.0, 0)
            cv2.putText(dd_sk, f'{er} mm', (round(0.38 * map_w), round(0.86 * map_h)), cv2.FONT_HERSHEY_PLAIN, fontSc,
                        (0, 255, 0), thickn, cv2.LINE_AA)

            perc = percentage
            percor = percentageor
            cv2.putText(dd, f'Undistrupted line: ', (round(0.05 * map_w), round(0.86 * map_h)), cv2.FONT_HERSHEY_PLAIN,
                        fontSc, (255, 255, 255), thickn, cv2.LINE_AA)
            cv2.putText(dd, f'{perc} %', (round(0.38 * map_w), round(0.86 * map_h)), cv2.FONT_HERSHEY_PLAIN, fontSc,
                        (0, 255, 0), thickn, cv2.LINE_AA)
            cv2.putText(dd, f'{percor} %', (round(0.55 * map_w), round(0.86 * map_h)), cv2.FONT_HERSHEY_PLAIN, fontSc,
                        (0, 0, 255), thickn, cv2.LINE_AA)
        elif cnt > th:
            FLerror = Ferror(RF_grey, Q_grey, circle, ref_wdth)

            dd_sk = draw_dynamic_texts(dd_sk, static_text_img, er, FLerror, fontSc, thickn)

            cv2.putText(dd, f'Undistrupted line: ', (round(0.05 * map_w), round(0.86 * map_h)), cv2.FONT_HERSHEY_PLAIN,
                        fontSc, (255, 255, 255), thickn, cv2.LINE_AA)
            cv2.putText(dd, f'{perc} %', (round(0.38 * map_w), round(0.86 * map_h)), cv2.FONT_HERSHEY_PLAIN, fontSc,
                        (0, 255, 0), thickn, cv2.LINE_AA)
            cv2.putText(dd, f'{percor} %', (round(0.55 * map_w), round(0.86 * map_h)), cv2.FONT_HERSHEY_PLAIN, fontSc,
                        (0, 0, 255), thickn, cv2.LINE_AA)

            cv2.putText(dd, 'Final path: ', (round(0.48 * map_w), round(0.15 * map_h)), cv2.FONT_HERSHEY_PLAIN, fontSc,
                        (255, 255, 255), thickn, cv2.LINE_AA)
            cv2.putText(dd, f'{percentage} %', (round(0.68 * map_w), round(0.15 * map_h)), cv2.FONT_HERSHEY_PLAIN,
                        fontSc, (0, 255, 0), thickn, cv2.LINE_AA)
            cv2.putText(dd, f'{percentageor} %', (round(0.78 * map_w), round(0.15 * map_h)), cv2.FONT_HERSHEY_PLAIN,
                        fontSc, (0, 0, 255), thickn, cv2.LINE_AA)

    canvas_all[:sz[0], :sz[1]] = original
    canvas_all[:sz[0], sz[1]:sz[1] * 2] = frame  # dst
    canvas_all[:sz[0], sz[1] * 2:] = ink  # dst#sb#mm
    canvas_all[sz[0]:, :int(sz[1] * 3 / 2)] = dd
    canvas_all[sz[0]:, int(sz[1] * 3 / 2):] = dd_sk
    # Ca = canvas_all[:,:,[2,1,0]]
    vout.write(canvas_all)

    cv2.imshow('Frame', canvas_all)
    # print(canvas_all.shape)
    cv2.waitKey(1)

    ul, fl, mp = [np.zeros((map_h, int(resolution_w * 3 / 2), 3)).astype(np.uint8) for _ in range(3)]
    RF_r = [np.zeros((map_h, int(resolution_w * 3 / 2), 3)).astype(np.uint8) for _ in range(1)]
    end_t = time.time()
    looptime.append(round(end_t - start_t, 5))
    avlpt=sum(looptime) / len(looptime)
    print('Loop time:', avlpt)

# cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()

picam2.stop_recording()
time.sleep(1)
vout.release()

gcode("G91")
gcode("G1 E-1 F400")
gcode("G90")
gcode("G1 Z25 F1000")
gcode("G1 X0 Y120 F1600")
# gcode("M18")
print("Print finished")





