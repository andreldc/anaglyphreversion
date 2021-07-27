import numpy as np
import cv2 as cv
import reciprocity
import utils

from config import REFINEMENT_CONFIG, SHOW_RESULTS, SHOW_OFF_MODE

WAITKEY_TIME = 1
if SHOW_OFF_MODE:
    WAITKEY_TIME = 0

def get_refinement(l_valid_disparity, r_valid_disparity, l_reciprocity, r_reciprocity):
    """
    Refines the initial disparity with a closing morphological operator
    """

    k_size = REFINEMENT_CONFIG["k_size"]

    # Defines the Kernel used for closing operation - O(1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))

    # Perform Closing Operation - O(1)
    l_closed_disparity = cv.morphologyEx(l_valid_disparity.astype("uint8"), cv.MORPH_CLOSE, kernel)
    r_closed_disparity = cv.morphologyEx(r_valid_disparity.astype("uint8"), cv.MORPH_CLOSE, kernel)

    # Substitute 0 with values found on closing operation - O(1)
    l_both_disparity = np.where(l_reciprocity == 0, l_closed_disparity, l_valid_disparity)
    r_both_disparity = np.where(r_reciprocity == 0, r_closed_disparity, r_valid_disparity)

    cv.imwrite("./l_both_disparity.jpg", utils.convert_to_image(l_both_disparity))
    cv.imwrite("./r_both_disparity.jpg", utils.convert_to_image(r_both_disparity))    

    # Aggregated Reciprocity Mask - O(N)
    l_both_disparity_valid, r_both_disparity_valid, l_both_reciprocity, r_both_reciprocity = \
        reciprocity.get_reciprocity(l_both_disparity, r_both_disparity, prevent_result_override=True)

    if SHOW_RESULTS:
        cv.imshow("Valid Disparity Map Closed", utils.stack_2_by_2(l_both_disparity_valid, \
            r_both_disparity_valid, l_both_reciprocity, r_both_reciprocity, convert_image=True))
        cv.imshow("Disparity Map Closed", utils.stack_2_by_2( \
            l_closed_disparity, r_closed_disparity, l_both_disparity, r_both_disparity, \
            convert_image=True))

        cv.waitKey(WAITKEY_TIME)

    return l_both_disparity_valid, r_both_disparity_valid, l_both_reciprocity, r_both_reciprocity
