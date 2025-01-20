from dataprep import *
from video_output_utils import *
from processing import *

def main():
    global circle, newfrm, newink, LK, LK_TM, temp_count, ul, fl, prv, prvpp, prv_skl, values1, values2, msk, perc, percor, er, dst_old, LK_TMold, last_mask_centerline, flag
    # Create necessary folders
    create_folders(newpath, ['Nozzle', 'Steps', 'Mapping', 'Mapping/Final', 'Mapping/Undstr', 'LK'])

    # Open video capture, set properties, read frame, and resize if requested
    cap = cv2.VideoCapture(SourceF + '/' + tname + '.mp4')
    _, frame = cap.read()
    frame, _ = image_resize(frame, vid_res)
    sz = (frame.shape[1], frame.shape[0])
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    vout, multiwindow_frame = initialize_video_writer(newpath, tname, sz, fps)

    # Start counting frames
    cnt = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cnt < length - frame_skip - 1:
        if cnt >= Thresholds_frames.dtct_r[0] and cnt <= Thresholds_frames.dtct_r[1]:
            _, frame = cap.read()
            cnt += 1
        else:
            for i in range(frame_skip+1):
                _, frame = cap.read()
                cnt += 1
        frame, _ = image_resize(frame, vid_res)
        original = frame.copy()
        cv2.putText(original, f'Frame #: {cnt}', (round(0.1 * vid_res), round(0.2 * vids_dimensions[3])),
                    cv2.FONT_HERSHEY_PLAIN, TextStyle.font[0], (0, 255, 0), TextStyle.thickness[0], cv2.LINE_AA)
        static_title_img = pre_render_static_title(TextStyle, vids_dimensions, vid_res)
        static_text_img = pre_render_static_text(cnt, Thresholds_frames.und_l[1], TextStyle, vids_dimensions, vid_res)

        # Enhance frame's contrast
        frame = contrast_stretch(frame)

        # Perform Hough Transform circle detection within specified range
        if cnt >= Thresholds_frames.dtct_r[0] and cnt < Thresholds_frames.dtct_r[1]:
            thresholded_mask = color_thresholding(frame, 'black', color_thresholds)
            circle, im, tmp = circle_detection(frame, thresholded_mask, TextStyle.thickness[2])

            # If a circle is detected, store the circle's radius and keep a snapshot
            if circle is not None:

                detected_circles_radius.append(circle[2])
                circle[2] = np.average(detected_circles_radius)
                tmp3c = cv2.merge((tmp, tmp, tmp))
                Detection_snap = np.concatenate((tmp3c, im), axis=1)
                cv2.imwrite(newpath + '/Nozzle/Nozzle detection example ' + str(cnt) + '_msk.jpg', Detection_snap)

                # Scale line thickness in Area accuracy
                TextStyle.line_thickness[0]=int(inkwidth_trg / (ref_wdth / (2 * circle[2])) * ratio)
            else:
                # If no circle is detected
                circle = np.array([0, 0, 0])

        ###########
        # Estimating ink's width
        Awl, processed_frame, ink_tmp = inks_width(
            frame, circle, tname, ref_wdth, cnt, Thresholds_frames, Dye_selected, color_thresholds,
            vids_dimensions, Aw, TextStyle.font[2], TextStyle.thickness[2], newpath)

        # Stitching ink using Lukas-Kanade optical flow
        LK, LK_TM, temp_count, dst_old, LK_TMold, ul, fl, values1, values2, newfrm, oldfrm, newink, flag = ink_stitching(
            LK, LK_TM, dst_old, newfrm, cnt, Thresholds_frames, flag, temp_count,
            newink, Zerofill, LK_TMold, tmul, tmfl, vids_dimensions, newpath, ul, fl, values1,
            values2, ink_tmp, Nsp, Nep, Feed_rates, TextStyle.line_thickness[1])

        # Create the area accuracy and centerline drift maps
        msk, mp, prv_skl, prv, prvpp, RF, RF_grey, RF_rc, last_mask_centerline = create_maps(prv_skl, prv, prvpp,
                                                                                   avheight, cnt, Thresholds_frames, TextStyle.line_thickness,
                                                                                   ul, fl, last_mask_centerline)
        # Calculate the area accuracy
        percentage, percentageor, Q_grey = compute_area_accuracy(RF_grey, mp, vids_dimensions, cnt, Thresholds_frames)

        # Calculate centerline drift and 'print' all metrics
        C_d, A_a, perc, percor, er = adding_accuracy_drift_results(cnt, Thresholds_frames, RF_grey, circle, ref_wdth, msk, RF_rc, Q_grey,
                                                       static_title_img, static_text_img, TextStyle.font[2], TextStyle.thickness[2], percentage, perc,
                                                       percentageor, percor, vids_dimensions, er)
        # Creating the frame of the video output
        multiwindow_frame = video_output(multiwindow_frame, sz, original, frame, processed_frame, A_a, C_d, vid_res)

        # Save captures that summarize the analysis
        snaps(cnt, length, original, A_a, C_d, processed_frame, frame, newpath, tname, Thresholds_frames)

        print(str(cnt) + str('/') + str(length - 1) + ' frame of file: ' + tname)

        # Write multi-window frame to output video
        vout.write(multiwindow_frame)

    # Close video capture and writer
    cap.release()
    vout.release()


if __name__ == "__main__":
    main()
