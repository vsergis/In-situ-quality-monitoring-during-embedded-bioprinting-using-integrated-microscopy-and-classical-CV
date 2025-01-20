from utils import*
from video_output_utils import*
import math

def circle_detection(image, mask, thickn, blur_iterations=5, threshold_black_pixel_ratio=0.5):
    """
    Combines two methods for detecting circles: Hough Circle Transform and color thresholding with pixel counting.

    Args:
        image (numpy.ndarray): The input image to process.
        Blower (numpy.ndarray): Lower bound for HSV thresholding (for Version 1).
        Bupper (numpy.ndarray): Upper bound for HSV thresholding (for Version 1).
        thickn : circle thickness
        blur_iterations (int): Number of times to apply median blur to the mask for color thresholding. Default is 5.
        lower_threshold (numpy.ndarray): Lower bound of the HSV threshold for Version 2.
        upper_threshold (numpy.ndarray): Upper bound of the HSV threshold for Version 2.
        threshold_black_pixel_ratio (float): Ratio of black pixels to total pixels that determines whether to skip circle detection. Default is 0.5.

    Returns:
        tuple: A tuple containing:
            - circles (numpy.ndarray or None): Detected circle's information [radius, x, y] or None if no circle is found.
            - mask (numpy.ndarray): The processed binary mask after thresholding and blurring (for Version 2).
    """

    # Step 1: Attempt Hough Circle Transform (Version 1)
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    mask_blurred = cv2.medianBlur(mask, 9)

    circles = cv2.HoughCircles(mask_blurred, cv2.HOUGH_GRADIENT, 8, 1, param1=130, param2=60)

    if circles is not None:
        # If circles are found, return the first one
        circle = circles[0, 0]
        x, y, r = map(int, circle)
        cv2.circle(image, (x, y), r, (0, 255, 0), thickn * 3)  # Draw the circle on the image
        circles = circle.astype(int)
        return circles, image, mask_blurred  # Return Hough Circle result
    else:
        # If no circle is detected, proceed to Version 2
        # Step 2: Attempt Color Thresholding and Pixel Counting (Version 2)
        #hsv_m = threshold_color(image, Dye_selected, color_thresholds)#cv2.inRange(hsv, lower_threshold, upper_threshold)

        #mask = hsv_m
        for _ in range(blur_iterations):
            mask = cv2.medianBlur(mask, 9)

        how_many_c = np.count_nonzero(mask, axis=0)  # Counts along columns (horizontal)
        how_many_r = np.count_nonzero(mask, axis=1)  # Counts along rows (vertical)

        # Check the number of non-zero (non-black) pixels
        total_pixels = mask.size
        black_pixel_count = np.sum(mask == 0)

        # If more than threshold_black_pixel_ratio of pixels are black, skip detection
        if black_pixel_count / total_pixels > threshold_black_pixel_ratio:
            return None, image, mask  # No circle detected due to excessive black pixels

        # Find the circle location and radius based on the maximum counts
        if np.max(how_many_c) > 0 and np.max(how_many_r) > 0:
            circles = np.array(
                [round((np.max(how_many_c) + np.max(how_many_r)) / 2), np.argmax(how_many_c), np.argmax(how_many_r)])
            cv2.circle(image, (circles[0], circles[1]), circles[2], (0, 0, 255), thickn * 3)  # Draw circle (green)
            return circles, image, mask  # Return color thresholding-based detection
        else:
            return None, image, mask  # No circle detected

def inks_width(frame, circle, tname, ref_wdth, cnt, tf, color_name, color_thresholds, vids_dimensions, Aw, font, thickness, newpath):
    """
    Process the width detection based on HSV color thresholds and display the results.

    Args:
    - frame: The current frame (image) from the video feed.
    - circle: The detected circle (used to filter width).
    - cnt: The current frame count.
    - th: Threshold frame count for displaying width.
    - lower1, upper1, lower2, upper2: HSV color thresholds for mask generation.
    - resolution_w, wdth_apprx: Resolution and approximate width for text placement.
    - Aw: List to store valid width values.
    - Fwidth: Function for calculating the width.
    - fontSc: Font scale for text.
    - thickn: Thickness for text display.
    - newpath: Path for saving output images.

    Returns:
    - Updated `Aw` list containing valid width values.
    - The processed frame with width text.
    """
    # Convert the frame to HSV color space
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks based on specified HSV ranges
    #lower_mask = cv2.inRange(hsv, lower1, upper1)
    #middle_mask = cv2.inRange(hsv, lower2, upper2)
    #upper_mask = cv2.inRange(hsv, lower2, upper2)
    #mask = lower_mask + middle_mask + upper_mask
    mask = color_thresholding(frame, color_name, color_thresholds)
    # Apply the mask to extract relevant parts of the image
    imask = mask > 0
    ink = np.zeros_like(frame, np.uint8)
    ink[imask] = frame[imask]  # Apply the mask to the frame
    ink_tmp = ink.copy()
    if circle.shape[0] > 2:  # Ensure that a valid circle is detected
        if cnt < tf.und_l[1]: # it was <= but it was giving an error, why?
            # Calculate the average width of the circle if the frame count is within threshold
            avwidthc = calculate_width(mask, circle, cnt, newpath, tf.und_l[1], tname, ref_wdth, vids_dimensions[4], vids_dimensions[3])
            cv2.putText(ink, f'Width: {avwidthc} mm',
                        (round(0.59 * vids_dimensions[4]), round(0.5 * vids_dimensions[3])),
                        cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 0), thickness, cv2.LINE_AA)

            # Append valid widths to the list if the width is less than 2 mm
            if not math.isnan(avwidthc) and avwidthc < 2:
                Aw.append(avwidthc)

        else:
            # If the frame count exceeds the threshold, display the average width
            avg_width = round(np.average(Aw), 3)
            cv2.putText(ink, f'Av W:{avg_width} mm',
                        (round(0.3 * vids_dimensions[4]), round(0.12 * vids_dimensions[3])),
                        cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 0), thickness, cv2.LINE_AA)

    return Aw, ink, ink_tmp  # Return the updated width list and the processed frame

def calculate_optflow(oldfrm,newfrm, cnt, newpath, resolution_w):
    newfrm=newfrm[:,round(0.1*resolution_w):round(1*resolution_w/2)]
    oldfrm=oldfrm[:,round(0.1*resolution_w):round(1*resolution_w/2)]
    listX, listY=[], []
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=10)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(120, 120), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #Random colors propably for the lines
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
    LK_X=sum(listX) / len(listX)
    LK_Y=sum(listY) / len(listY)

    cv2.imwrite(newpath+'/LK/LK '+str(cnt)+'.jpg', img)
    TM=np.float32([[1, 0, int(LK_X)], [0, 1, int(LK_Y)]])
    return TM

def ink_stitching(LK, LK_TM, dst_old, newfrm, cnt, tf, flag, temp_count,
                       newink, Zerofill, LK_TMold, tmul, tmfl, vids_dimensions,
                       newpath, ul, fl, values1, values2, ink_tmp, Nsp, Nep, feed_rates, line_thickness):
    """
    Processes the LK mapping and frame warping for a video frame. This function handles the transformation and tracking of frames based on LK mapping.
    tf: Thresholds_frames class map_w, mapd, resolution_w, wdth_apprx
    """
    ratio=[]
    oldfrm = newfrm
    oldink = newink
    newfrm = create_centerline(ink_tmp, 2, line_thickness)
    newink = ink_tmp.copy()
    newfrm[Nsp[0]:Nsp[1], Nep[0]:Nep[1]] = Zerofill
    # nextLK=cnt
    if oldfrm is not None and (cnt > tf.und_l[0] or cnt > tf.fnl_p[0]):
        if cnt >= tf.fnl_p[0] and flag == 0:
            dst_old = np.zeros_like(oldink)
            dst_pad = np.zeros_like(oldink)
            LK_TMupd = np.concatenate((LK_TM[:, :-1], np.zeros([2, 1])), axis=1)
            LK_TMold = np.concatenate((LK_TM[:, :-1], np.zeros([2, 1])), axis=1)
            flag = 1
        if cnt < tf.und_l[1] or (cnt >= tf.fnl_p[0] and cnt < tf.fnl_p[1]-3): #giati omws mono me -3?
            if cnt < tf.und_l[1]:
                LK_TM = calculate_optflow(oldfrm, newfrm, cnt, newpath, vids_dimensions[4])
                LK.append(LK_TM)
                lksm1 = sum(LK)[0][2]
                lkr1 = sum(LK)[0][0]
                lksm2 = sum(LK)[0][2] * ((feed_rates[0]/feed_rates[1]))#1.2  # *0.50#((F1/F2))#*0.4887218045112782
                lkr2 = sum(LK)[0][0] * (tf.fnl_p[1] - tf.fnl_p[0]) / (tf.und_l[1] - tf.und_l[0])
                base_value_1 = lksm1 // lkr1
                base_value_2 = lksm2 // lkr2
                # Calculate the remainder
                remainder_1 = lksm1 % lkr1
                remainder_2 = lksm2 % lkr2
                values1 = [base_value_1 + 1 if i < abs(int(remainder_1)) else base_value_1 for i in range(int(lkr1))]
                values2 = [base_value_2 + 1 if i < abs(int(remainder_2)) else base_value_2 for i in range(int(lkr2))]
            elif cnt >= tf.und_l[0]:
                LM = values2[temp_count[1]]
                temp_count[1] += 1
                temp_count[2] += 1
                LK_TM[0][2] = LM

            tm = LK_TM[:, -1] + LK_TMold[:, -1]
            tm = tm.reshape(2, 1)
            LK_TMupd = np.concatenate((LK_TM[:, :-1], tm), axis=1)
            dst_pad = warping_padding(oldink, newink, LK_TMupd)
            h = np.min([dst_pad.shape[0], oldink.shape[0]])

            if (LK_TM == LK_TMupd).all():
                dst_pad[:h, :abs(int(LK_TMupd[0][2]))] = oldink[:h, :abs(int(LK_TM[0][2]))]
            else:
                if int(LK_TMupd[1][2]) == int(LK_TMold[1][2]):
                    dst_pad[:h, abs(int(LK_TMold[0][2])):abs(int(LK_TMupd[0][2]))] = oldink[:h, :abs(int(LK_TM[0][2]))]
                    dst_pad[:h, :abs(int(LK_TMold[0][2]))] = dst_old[:h, :abs(int(LK_TMold[0][2]))]
                else:
                    dst_pad[:h, abs(int(LK_TMold[0][2])):abs(int(LK_TMupd[0][2]))] = oldink[:h, :abs(int(LK_TM[0][2]))]
                    dst_pad[:h, :abs(int(LK_TMold[0][2]))] = dst_old[:h, :abs(int(LK_TMold[0][2]))]
            if cnt < tf.und_l[1]:
                UL = cv2.warpAffine(dst_pad, tmul, (round(vids_dimensions[3] * vids_dimensions[2]), vids_dimensions[3]), borderValue=(0, 0, 0))
                ul, _ = image_resize(UL, width=vids_dimensions[0])
                cv2.imwrite(newpath + '/Mapping/Undstr/Mapping progress' + str(cnt) + '.jpg', ul)
            elif cnt > tf.fnl_p[0] and cnt < tf.fnl_p[1]:
                FL = cv2.warpAffine(dst_pad, tmfl, (round(vids_dimensions[3] * vids_dimensions[2]), vids_dimensions[3]), borderValue=(0, 0, 0))
                fl, _ = image_resize(FL, width=vids_dimensions[0])
                cv2.imwrite(newpath + '/Mapping/Final/Mapping progress' + str(cnt) + '.jpg', fl)
            tm = LK_TM[:, -1] + LK_TMold[:, -1]
            tm = tm.reshape(2, 1)
            LK_TMold = np.concatenate((LK_TM[:, :-1], tm), axis=1)
            dst_old = dst_pad.copy()

    return LK, LK_TM, temp_count, dst_old, LK_TMold, ul, fl, values1, values2, newfrm, oldfrm, newink, flag
def create_maps(last_ink_centerline, last_ink_area, prvpp, avheight, cnt, tf, line_thickness, ul, fl, last_mask_centerline):
    """
    Helper function to update masks and reference paths.
    """
    if cnt < tf.und_l[1]:
        ink_area = ul  # merge#[0]#cv2.addWeighted(merge[0], alpha, merge[1], 1 - alpha, 0)
        ink_centerline = create_centerline(ul, 0, line_thickness[1])  # merge_skel_#[0].copy()
        last_ink_area = ink_area  # merge#[0].copy()
        last_ink_centerline = ink_centerline  # merge_skel_#[0].copy()

    else:
        ink_area = fl  # merge#[1]
        ink_centerline = create_centerline(fl, 1, line_thickness[1])  # merge_skel_#[1].copy()



    mask_ink = ink_area != [0, 0, 0]
    # Color the pixels in the mask
    Target_ink, prvpp = add_target_path(ink_area.copy(), cnt, tf.und_l[1], prvpp, avheight, thickness=line_thickness[0]) # .astype(np.uint8)
    Target_ink_copy = Target_ink.copy()
    Target_ink_copy[mask_ink] = ink_area[mask_ink]

    Target_centerline, prvpp = add_target_path(ink_centerline.copy(), cnt, tf.und_l[1], prvpp, avheight, color=(255, 0, 0), thickness=line_thickness[1])
    mask_ink_centerline = ink_centerline != [0, 0, 0]
    Target_centerline_copy = Target_centerline.copy()
    Target_centerline_copy[mask_ink_centerline] = ink_centerline[mask_ink_centerline]

    RF_grey = cv2.cvtColor(Target_ink.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    RF = Target_centerline.copy()
    if cnt < tf.und_l[1]:
        Area_accuracy_map=Target_ink_copy
        Centerline_drift=Target_centerline_copy
        last_mask_centerline=mask_ink_centerline
    elif cnt >= tf.und_l[1]:
        Area_accuracy_map = cv2.addWeighted(last_ink_area, 0.6, Target_ink_copy, 1 - 0.6, 0)
        #Centerline_drift = cv2.addWeighted(last_ink_centerline, 0.6, Target_centerline_copy, 1 - 0.6, 0)
        Target_centerline_copy[last_mask_centerline] = last_ink_centerline[last_mask_centerline]
        Centerline_drift=Target_centerline_copy

    return Centerline_drift, ink_area, last_ink_centerline, last_ink_area, prvpp, RF, RF_grey, Area_accuracy_map, last_mask_centerline

def compute_area_accuracy(RF_grey, mp, maps_dimensions, cnt, tf):
    # Put the accuracy text on the map

    if cnt <= tf.und_l[0] or (cnt>tf.und_l[1] and cnt<tf.fnl_p[0]):
        return 0, 0, np.zeros((maps_dimensions[0], maps_dimensions[1], 3)).astype(np.uint8)  # No computation if the condition isn't met

    Q_grey = cv2.cvtColor(mp.copy(), cv2.COLOR_RGB2GRAY) #auto einai to ink san na leme
    _, Q_grey = cv2.threshold(Q_grey, 0, 255, cv2.THRESH_BINARY) # to kanoume aspromavro - binary
    _, RF_bin = cv2.threshold(RF_grey, 0, 255, cv2.THRESH_BINARY) # autos einai o stoxos - referemce kai to kanoume binary

    bitwiseand = cv2.bitwise_and(RF_grey, Q_grey) # edw tsekaroume ta idia pixel metaksi stoxou kai tou ink/pragmatikotitas
    _, bitwiseand = cv2.threshold(bitwiseand, 0, 255, cv2.THRESH_BINARY) # edw to kanoume binary gia sigouria

    bitwiseor = RF_bin - Q_grey # edw apla aferoume ink apo reference kai perisevei oti einai apeksw

    # Calculate the total pixels in the RF_grey and Q_grey area
    total = RF_bin
    total_pixels = np.count_nonzero(total)  # edw metrame posa pixels einai o stoxos
    #total_ink = np.count_nonzero(Q_grey)  # This replaces the previous loop checking for non-zero pixels

    if total_pixels > 0:# and total_ink > 0:
        # Matches
        matches = np.count_nonzero(bitwiseand)
        matchesor = np.count_nonzero(bitwiseor)

        # Calculate percentage
        percentage = round((100 * matches / total_pixels), 2)
        percentageor = round((100 * matchesor / total_pixels), 2)
    else:
        percentage = 0
        percentageor = 0

    return percentage, percentageor, Q_grey

def adding_accuracy_drift_results(cnt, tf, RF_grey, circle, ref_wdth, dd_sk, dd, Q_grey, static_title_img, static_text_img, font, thickness, percentage, perc,
                     percentageor, percor, maps_dimensions, er):
    """
    Main function to process accuracy calculations and update the images accordingly.
    """
    dd = cv2.addWeighted(dd, 1.0, static_title_img[0], 1.0, 0)
    dd_sk = cv2.addWeighted(dd_sk, 1.0, static_title_img[1], 1.0, 0)
    if cnt >= tf.und_l[0]:
        if cnt < tf.und_l[1]:
            ULerror = calculate_drift(cnt, RF_grey, Q_grey, circle, ref_wdth, maps_dimensions)
            if ULerror is not None:
                er = ULerror.copy()
            else:
                er = None
            dd = cv2.addWeighted(dd, 1.0, static_text_img, 1.0, 0)
            dd_sk = cv2.addWeighted(dd_sk, 1.0, static_text_img, 1.0, 0)
            cv2.putText(dd_sk, f'{er} mm', (round(0.3 * maps_dimensions[0]), round(0.86 * maps_dimensions[1])), cv2.FONT_HERSHEY_PLAIN, font,
                        (0, 0, 255), thickness, cv2.LINE_AA)

            perc = percentage
            percor = percentageor
            dd = draw_dynamic_textsAcc(cnt, tf.und_l[1], dd, perc, percor, percentage, percentageor, font, thickness, maps_dimensions)
        elif cnt > tf.und_l[1]:
            FLerror = calculate_drift(cnt, RF_grey, Q_grey, circle, ref_wdth, maps_dimensions)

            dd = cv2.addWeighted(dd, 1.0, static_text_img, 1.0, 0)

            dd_sk = draw_dynamic_texts(dd_sk, static_text_img, er, FLerror, font, thickness, maps_dimensions)
            dd = draw_dynamic_textsAcc(cnt, tf.und_l[1], dd, perc, percor, percentage, percentageor, font, thickness, maps_dimensions)

    return dd_sk, dd, perc, percor, er  # Return the updated images
