from sys import stderr
import numpy as np
import cv2 as cv
import utils
from config import RECIPROCITY_CONFIG, SHOW_RESULTS, SHOW_OFF_MODE

WAITKEY_TIME = 1
if SHOW_OFF_MODE:
    WAITKEY_TIME = 0

def get_reciprocity(l_disparity_map, r_disparity_map, scale_factor=1, prevent_result_override=False):
    """
    Computes reciprocity between left and right disparity map

    Args:
        l_disparity_map ([type]): [description]
        r_disparity_map ([type]): [description]
        scale_factor (int, optional): [description]. Defaults to 1.
        prevent_result_override (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    threshold = RECIPROCITY_CONFIG["threshold"]

    # If disparity map is out of scale, re-scales it - O(1)
    l_disparity_map_resized = np.round(l_disparity_map / scale_factor).astype('int16')
    r_disparity_map_resized = np.round(r_disparity_map / scale_factor).astype('int16')

    y_axis, x_axis = l_disparity_map_resized.shape

    # Variables to store Mask - O(1)
    l_reciprocity = np.zeros((y_axis, x_axis), 'int16')
    r_reciprocity = np.zeros((y_axis, x_axis), 'int16')

    # For each pixel p in the image - O(2.N)
    for y in range(y_axis):
        utils.progress_bar(y, y_axis, text="Reciprocity Calculation")
        for x in range(x_axis):

            # Find the disparity for p in the LEFT Image - O(1)
            l_disparity = l_disparity_map_resized[y, x]

            # Finds the correspondent x2 position in the RIGHT image - O(1)
            l_x2 = x - l_disparity

            # If the x2 position is within RIGHT image dimensions - O(1)
            if l_x2 > 0 and l_x2 < x_axis:
                # Find the disparity for x2
                r_disparity_from_l = r_disparity_map_resized[y, l_x2]
            else:
                r_disparity_from_l = -10000

            # If both disparities are greater than 0
            # AND their difference is within a given limit - O(1)
            if l_disparity > 0 and r_disparity_from_l > 0 \
               and abs(l_disparity - r_disparity_from_l) <= threshold:
                # Sets p as a valid pixel
                l_reciprocity[y, x] = 1
            else:
                # Sets p as a invlid pixel
                l_reciprocity[y, x] = 0

            # Find the disparity for p in the RIGHT Image - O(1)
            r_disparity = r_disparity_map_resized[y, x]

            # Finds the correspondent x2 position in the LEFT image - O(1)
            r_x2 = x + r_disparity

            # If the x2 position is within LEFT image dimensions - O(1)
            if r_x2 > 0 and r_x2 < x_axis:
                # Find the disparity for x2
                l_disparity_from_r = l_disparity_map_resized[y, r_x2]
            else:
                l_disparity_from_r = -10000

            # If both disparities are greater than 0
            # AND their difference is within a given limit - O(1)
            if r_disparity > 0 and l_disparity_from_r > 0 \
               and abs(r_disparity - l_disparity_from_r) <= threshold:
                # Sets p as a valid pixel
                r_reciprocity[y, x] = 1
            else:
                # Sets p as a invlid pixel
                r_reciprocity[y, x] = 0


    # Computes valid disparity Maps - O(1)
    l_valid_disparity_map = np.where(l_reciprocity == 1, l_disparity_map_resized, 0)
    r_valid_disparity_map = np.where(r_reciprocity == 1, r_disparity_map_resized, 0)

    if SHOW_RESULTS and not prevent_result_override:
        cv.imshow("Valid Disparity Map", utils.stack_2_by_2(l_valid_disparity_map, \
            r_valid_disparity_map, l_reciprocity, r_reciprocity, convert_image=True))
        cv.waitKey(WAITKEY_TIME)

    return l_valid_disparity_map, r_valid_disparity_map, l_reciprocity, r_reciprocity
