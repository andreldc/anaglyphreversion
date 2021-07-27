import numpy as np
import cv2 as cv
import utils
from config import SHOW_RESULTS, SHOW_OFF_MODE

WAITKEY_TIME = 1
if SHOW_OFF_MODE:
    WAITKEY_TIME = 0

def get_disparity_map(l_costs, r_costs):

    # Gets disparity map from Data Costs Volume - O(1)
    l_disparity_map = np.argmin(l_costs, 0)
    r_disparity_map = np.argmin(r_costs, 0)

    if SHOW_RESULTS:
        cv.imshow("Disparity Map", 
                  utils.stack_1_by_2(l_disparity_map, r_disparity_map, convert_image=True))
        cv.waitKey(WAITKEY_TIME)

    return l_disparity_map, r_disparity_map
