import numpy as np
import cv2 as cv
from utils import stack_1_by_2, stack_1_by_3, label_image
from config import ANAGLYPH_CONFIG, SHOW_RESULTS, SHOW_OFF_MODE

WAITKEY_TIME = 1
if SHOW_OFF_MODE:
    WAITKEY_TIME = 0

def get_anaglyph_channels(anaglyph, avoid_image_show=False):
    """
    Maps anaglyph to Left and Right channels

    Parameters:
    anaglyph (uint8 np.array): Anaglyph image
    right_left_factor (float): Factor for merging Right channel image

    Returns:
    l_channel, r_channel, r1_channel, r2_channel (uint8 cv2 image np.array):
    l_channel is the left channel
    r_channel is the merged right channel
    r1_channel and r2_channel are the right channel

    Requirements:
    """
    blue, green, red = cv.split(anaglyph)

    if ANAGLYPH_CONFIG["color_scheme"] == 'gm':
        l_channel = green
        r1_channel = red
        r2_channel = blue

        l_channel_label = "green"
        r1_channel_label = "red"
        r2_channel_label = "blue"
        r_channel_label = "magenta"
    elif ANAGLYPH_CONFIG["color_scheme"] == 'by':
        l_channel = blue
        r1_channel = red
        r2_channel = green

        l_channel_label = "blue"
        r1_channel_label = "red"
        r2_channel_label = "green"
        r_channel_label = "yellow"
    else:
        l_channel = red
        r1_channel = green
        r2_channel = blue

        l_channel_label = "red"
        r1_channel_label = "green"
        r2_channel_label = "blue"
        r_channel_label = "cyan"

    r_channel = np.rint(r1_channel * ANAGLYPH_CONFIG["right_left_factor"] +
                        r2_channel * (1 - ANAGLYPH_CONFIG["right_left_factor"])
                        ).astype("uint8")

    if SHOW_RESULTS and not avoid_image_show:
        cv.imshow("Anaglyph Input Image", anaglyph)
        cv.waitKey(WAITKEY_TIME)

        l_channel_labeled = label_image(l_channel, l_channel_label, color=[255, 255, 255])
        r1_channel_labeled = label_image(r2_channel, r1_channel_label, color=[255, 255, 255])
        r2_channel_labeled = label_image(r2_channel, r2_channel_label, color=[255, 255, 255])
        r_channel_labeled = label_image(r_channel, r_channel_label, color=[255, 255, 255])

        cv.imshow("Anaglyph Channels", stack_1_by_3(l_channel_labeled, r1_channel_labeled,
                                                    r2_channel_labeled))
        cv.waitKey(WAITKEY_TIME)

        empty = np.zeros(l_channel.shape).astype('uint8')

        blue_color = cv.merge((blue, empty, empty))
        green_color = cv.merge((empty, green, empty))
        red_color = cv.merge((empty, empty, red))
        cyan_color = cv.merge((blue, green, empty))

        cv.imshow("Anaglyph Channels (Color representation)", stack_1_by_3(red_color, green_color,
                                                                           blue_color))
        cv.waitKey(WAITKEY_TIME)

        cv.imshow("Left red x merged right cyan", stack_1_by_2(l_channel_labeled,
                                                               r_channel_labeled))
        cv.waitKey(WAITKEY_TIME)

        cv.imshow("Left red x merged right cyan (Color representation)",
                  stack_1_by_2(red_color, cyan_color))
        cv.waitKey(WAITKEY_TIME)

    return (l_channel, r_channel, r1_channel, r2_channel)
