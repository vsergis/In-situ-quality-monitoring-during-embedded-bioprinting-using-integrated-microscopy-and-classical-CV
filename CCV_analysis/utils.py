import cv2
import numpy as np

def contrast_stretch(frame, min_percent=1, max_percent=95):
    """
    Apply contrast stretching to the frame based on the provided percentiles.
    """
    lo, hi = np.percentile(frame, (min_percent, max_percent))
    res_img = (frame.astype(float) - lo) / (hi - lo)
    return np.maximum(np.minimum(res_img * 255, 255), 0).astype(np.uint8)

def color_thresholding(image, color_name, color_thresholds):
    # Get the threshold ranges for the selected color
    color_ranges = color_thresholds.get(color_name, [])

    if not color_ranges:
        raise ValueError(f"No thresholds found for color: {color_name}")

    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = np.zeros_like(hsv_image[:, :, 0])  # Create a blank mask

    # Loop through the ranges and apply the thresholding
    for color_range in color_ranges:
        lower = color_range['lower']
        upper = color_range['upper']

        # Apply the range thresholding
        current_mask = cv2.inRange(hsv_image, lower, upper)

        # Combine with the existing mask (OR operation)
        mask = cv2.bitwise_or(mask, current_mask)

    return mask

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes an image to a given width or height, maintaining the aspect ratio.

    Args:
        image (numpy.ndarray): The input image to be resized.
        width (int, optional): The target width for resizing. Defaults to None.
        height (int, optional): The target height for resizing. Defaults to None.
        inter (cv2 interpolation method, optional): The interpolation method to use. Defaults to cv2.INTER_AREA.

    Returns:
        resized (numpy.ndarray): The resized image.

    Raises:
        ValueError: If both width and height are None.
    """
    # Check if both width and height are None
    if width is None and height is None:
        raise ValueError("At least one of width or height must be provided.")

    # Get the original dimensions of the image
    (h, w) = image.shape[:2]

    # Initialize the target dimensions
    dim = None

    # If height is not provided, compute from width
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    # If width is not provided, compute from height
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image using the calculated dimensions
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized, r

def add_target_path(Limg, cnt, threshold, prvpp, avheight, color=(255, 0, 0), thickness=2):

    positions = np.nonzero(Limg.T)
    if cnt <= threshold:
        if not positions[1].any():
            ref = np.zeros_like(Limg)
        else:
            avheight.append(round((min(positions[2])+max(positions[2]))/2))
            p1=(min(positions[1]),round(np.average(avheight)*1.03))
            p2=(max(positions[1]),round(np.average(avheight)*1.03))
            scl = (p2[0] - p1[0]) / 18
            ref = cv2.line(np.zeros_like(Limg), p1, p2, color, thickness)
            prvpp=[p1, p2, scl]
    else:
        p3 = tuple(np.subtract(prvpp[1], (0, round(1.7 * prvpp[2]))))
        p4 = tuple(np.subtract(p3, (round(3 * prvpp[2]), 0)))
        p5 = tuple(map(sum, zip(p4, (0, round(0.5 * prvpp[2])))))
        p6 = tuple(np.subtract(p5, (round(3 * prvpp[2]), 0)))
        p7 = tuple(map(sum, zip(p6, (0, round(0.5 * prvpp[2])))))
        p8 = tuple(np.subtract(p7, (round(3 * prvpp[2]), 0)))

        ref = cv2.line(np.zeros_like(Limg), prvpp[0], prvpp[1], color, thickness)
        ref = cv2.line(ref, prvpp[1], p3, color, thickness)
        ref = cv2.line(ref, p3, p4, color, thickness)
        ref = cv2.line(ref, p4, p5, color, thickness)
        ref = cv2.line(ref, p5, p6, color,thickness) #-168
        ref = cv2.line(ref, p6, p7, color,thickness) #-28
        ref = cv2.line(ref, p7, p8, color,thickness) #-168
    return ref, prvpp

def uniqueX_averageY(p):
    unique_positions, indices = np.unique(p[0], return_inverse=True)
    sums = np.zeros_like(unique_positions, dtype=np.float64)
    counts = np.zeros_like(unique_positions, dtype=np.int32)
    np.add.at(sums, indices, p[1])
    np.add.at(counts, indices, 1)
    averages = sums / counts
    return np.column_stack((unique_positions, averages)).reshape(-1, 2)

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

def create_centerline(cmp, segment, line_thickness):
    if len(cmp.shape)>2:
        img = cv2.cvtColor(cmp, cv2.COLOR_BGR2GRAY)
    else:
        img=cmp
    _, gray = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(gray, (5, 5), 0)
    line = cv2.ximgproc.thinning(img)
    if segment == 0:
        line = cv2.merge((np.zeros_like(line), np.zeros_like(line), line))
    elif segment == 1:
        line = cv2.merge((np.zeros_like(line), line, np.zeros_like(line)))

    kernel_size = line_thickness  # You can adjust this value for thicker edges
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation to increase the thickness of the edges
    dilated_image = cv2.dilate(line, kernel, iterations=1)

    return dilated_image

def warping_padding(src, dst, transf):
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

def calculate_width(image, circle, cnt, newpath, threshold, tname, ref_wdth, resolution_w, wdth_apprx):
        x, y, r = circle[:]  # Simplify tuple unpacking
        # Extract the region of interest from the image
        img = image[y - round(0.21 * wdth_apprx):y + round(0.21 * wdth_apprx),
              x - round(0.2578 * resolution_w):x - round(0.0625 * resolution_w)]
        # Optionally, save the extracted region as an image
        if cnt == threshold:
            cv2.imwrite(f"{newpath}/Mask_example_{tname[0]}.jpg", img)
        # Count the non-zero pixels along the columns
        how_many_c = np.count_nonzero(img, axis=0)
        # Calculate the average width
        real_hmc = how_many_c[how_many_c != 0]
        avwidthc = np.average(real_hmc / (2 * r)) * ref_wdth
        #if cnt>=115:
        #    print(['ok'])
        return round(avwidthc, 3)  # Round only when returning the result

def calculate_drift(cnt, reference, image, circle, ref_wdth, maps_dimensions):
    drift = []

    # Segmentation
    hl1_r, hl1_i = reference[round(0.39*maps_dimensions[1]):, :round(0.84*maps_dimensions[0])], image[round(0.39*maps_dimensions[1]):, :round(0.84*maps_dimensions[0])]
    hl234_r, hl234_i = reference[:round(0.39*maps_dimensions[1]), :round(0.84*maps_dimensions[0])], image[:round(0.39*maps_dimensions[1]), :round(0.84*maps_dimensions[0])]
    vl1_r, vl1_i = reference[:, round(0.84*maps_dimensions[0]):], image[:, round(0.84*maps_dimensions[0]):]

    Segm = [(hl1_r, hl1_i), (hl234_r, hl234_i), (vl1_r, vl1_i)]
    for i, (seg_r, seg_i) in enumerate(Segm):
        # Non-zero positions
        positions_ref = np.nonzero(seg_r.T) if i < 2 else np.nonzero(seg_r)
        positions_img = np.nonzero(seg_i.T) if i < 2 else np.nonzero(seg_i)

        if not positions_img[0].any() or not positions_ref[0].any():
            continue

        # Calculate averages
        res_r = uniqueX_averageY(positions_ref)
        res_i = uniqueX_averageY(positions_img)

        # Find common x-values and extract corresponding y-values from both arrays
        common_x = np.intersect1d(res_r[:, 0], res_i[:, 0])
        drift_segment = np.mean(res_r[np.isin(res_r[:, 0], common_x), 1] - res_i[np.isin(res_i[:, 0], common_x), 1])
        drift.append(drift_segment)

    r = circle[2]
    if not drift:
        error = None
    else:
        drift=[x for x in drift if str(x) != 'nan']
        error = round((sum(map(abs, drift)) / len(drift) / (2 * r)) * ref_wdth, 3)
    return error
