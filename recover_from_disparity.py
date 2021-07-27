import numpy as np
import cv2 as cv
from split_anaglyph_channels import get_anaglyph_channels
import utils

from config import SHOW_RESULTS, SHOW_OFF_MODE

WAITKEY_TIME = 1
if SHOW_OFF_MODE:
    WAITKEY_TIME = 0

def recover(anaglyph, l_disparity_map, r_disparity_map):

    y_axis, x_axis, _ = anaglyph.shape

    # Opens anaglyph and splits its channels = O(1)
    l_channel, _, r1_channel, r2_channel = get_anaglyph_channels(anaglyph, avoid_image_show=True)

    left_red = l_channel
    right_green = r1_channel
    right_blue = r2_channel

    left_green = np.zeros((y_axis, x_axis), 'uint8')
    left_blue = np.zeros((y_axis, x_axis), 'uint8')
    right_red = np.zeros((y_axis, x_axis), 'uint8')

    # For each pixel p - O(N)
    for y in range(y_axis):
        for x in range(x_axis):

            # Gets the disparity - O(1)
            l_disparity = l_disparity_map[y, x]
            r_disparity = r_disparity_map[y, x]

            # If the disparity leads to a pixel within the image - O(1)
            if x - l_disparity > 0:
                # Copies color
                left_green[y, x] = right_green[y, int(x - l_disparity)]
                left_blue[y, x] = right_blue[y, int(x - l_disparity)]

            # If the disparity leads to a pixel within the image - O(1)
            if x + r_disparity < x_axis:
                # Copies color
                right_red[y, x] = left_red[y, int(x + r_disparity)]

    # Gets valid mask from disparity map - O(1)
    l_valid_mask = np.where(l_disparity_map != 0, 1, 0)
    r_valid_mask = np.where(r_disparity_map != 0, 1, 0)

    l_recovered = cv.merge((
        np.where(l_valid_mask == 1, left_blue, 0),
        np.where(l_valid_mask == 1, left_green, 0),
        np.where(l_valid_mask == 1, left_red, 0)
        )).astype("uint8")
    r_recovered = cv.merge((
        np.where(r_valid_mask == 1, right_blue, 0),
        np.where(r_valid_mask == 1, right_green, 0),
        np.where(r_valid_mask == 1, right_red, 0)
        )).astype("uint8")

    if SHOW_RESULTS:
        l_disparity_3c = cv.merge((l_disparity_map, l_disparity_map, l_disparity_map))
        r_disparity_3c = cv.merge((r_disparity_map, r_disparity_map, r_disparity_map))

        cv.imshow("Recovered", utils.stack_2_by_2(l_disparity_3c, r_disparity_3c,\
                                                  l_recovered, r_recovered, convert_image=True))
        cv.waitKey(WAITKEY_TIME)

    return l_recovered, r_recovered
