import numpy as np
import cv2 as cv
import utils
from split_anaglyph_channels import get_anaglyph_channels
from config import COLORIZATION_CONFIG, SHOW_RESULTS

def erode(l_reciprocity_mask_temp, r_reciprocity_mask_temp):
    erosion_kernel = COLORIZATION_CONFIG["erosion_kernel"]
    kernel = np.ones((erosion_kernel, erosion_kernel),np.uint8)

    l_eroded = cv.erode(l_reciprocity_mask_temp, kernel, iterations = 1)
    r_eroded = cv.erode(r_reciprocity_mask_temp, kernel, iterations = 1)

    if SHOW_RESULTS:
        cv.imshow("Eroded reciprocity",
                  utils.stack_1_by_2(utils.convert_to_image(l_reciprocity_mask_temp),
                  utils.convert_to_image(r_reciprocity_mask_temp)))
        cv.waitKey(1)

    return l_eroded, r_eroded


def get_weights(block):

    block = block.astype('int16')
    center_index = int((block.shape[0]-1)/2)
    center_value = block[center_index, center_index]
    delta_color = np.absolute(block - center_value)

    return delta_color


def get_block(image, x, y, window_size):
    ''' Get a block from a given image at coordinates x,y '''
    half_window_size = int(window_size/2)

    bordered = cv.copyMakeBorder(image, half_window_size, half_window_size,
                                 half_window_size, half_window_size, cv.BORDER_REFLECT_101)

    x_start = x
    x_end = x_start + window_size
    y_start = y
    y_end = y_start + window_size

    return bordered[y_start:y_end, x_start:x_end]

def get_invalid_borders(reciprocity):
    """
    Get the list of invalid pixels that are in the borders
    """
    l_trim = np.zeros(reciprocity.shape)
    l_trim[:, :-1] = reciprocity[:, 1:]

    r_trim = np.zeros(reciprocity.shape)
    r_trim[:, 1:] = reciprocity[:, :-1]

    u_trim = np.zeros(reciprocity.shape)
    u_trim[:-1, :] = reciprocity[1:, :]

    b_trim = np.zeros(reciprocity.shape)
    b_trim[1:, :] = reciprocity[:-1, :]

    diff1 = (l_trim - reciprocity)
    diff2 = (r_trim - reciprocity)
    diff3 = (u_trim - reciprocity)
    diff4 = (b_trim - reciprocity)

    borders = np.where(diff1 > 0, 1, 0)
    borders = np.where(diff2 > 0, 1, borders)
    borders = np.where(diff3 > 0, 1, borders)
    borders = np.where(diff4 > 0, 1, borders)

    if SHOW_RESULTS:
        cv.imshow("Borders (Colorization)", utils.convert_to_image(borders))
        cv.waitKey(1)

    invalid = [np.where(borders == 1)][0]

    return invalid

def colorize(anaglyph, l_recovered, r_recovered, l_reciprocity_mask, r_reciprocity_mask):

    y_axis, x_axis, _ = anaglyph.shape
    img_area = y_axis*x_axis
    scale_factor = img_area/166080

    # Scales factors to adjust to image size - O(1)
    min_matches      = int(scale_factor * COLORIZATION_CONFIG["min_matches"])
    max_window_size  = int(y_axis/4)*2 - 1
    min_window_size  = int(scale_factor * COLORIZATION_CONFIG["min_window_size"]/2)*2 + 1
    window_increment = int(scale_factor * COLORIZATION_CONFIG["window_increment"]/2)*2 + 1

    # Get anaglyph Channels - O(1)
    red_channel, cyan_channel, _, _ = get_anaglyph_channels(anaglyph, avoid_image_show=True)

    # Variable to store final image - O(1)
    l_colorized = np.copy(l_recovered)
    r_colorized = np.copy(r_recovered)

    # Define temporary reciprocity mask to update iteratively - O(1)
    l_reciprocity_mask_temp = np.copy(l_reciprocity_mask)
    r_reciprocity_mask_temp = np.copy(r_reciprocity_mask)

    # Erodes reciprocity masks = enhanced results
    l_reciprocity_mask_temp, r_reciprocity_mask_temp = erode(l_reciprocity_mask_temp,
                                                             r_reciprocity_mask_temp)

    # Array to store actual window size for each colorized pixel
    l_window_sizes = (np.ones((y_axis, x_axis))*min_window_size).astype('int16')
    r_window_sizes = (np.ones((y_axis, x_axis))*min_window_size).astype('int16')

    l_cut_thds = (np.ones((y_axis, x_axis))*COLORIZATION_CONFIG["threshold"]).astype('int16')
    r_cut_thds = (np.ones((y_axis, x_axis))*COLORIZATION_CONFIG["threshold"]).astype('int16')

    l_invalid_count = utils.count_if(l_reciprocity_mask_temp, 0)
    r_invalid_count = utils.count_if(r_reciprocity_mask_temp, 0)

    # While there are invalid pixels - O(I.) - I=Invalid pixels, t=Tries (max-min/inc)
    while l_invalid_count > 0:

        # Get invalid borders
        l_invalid = get_invalid_borders(l_reciprocity_mask_temp)

        # For each invalid pixel
        for i in range(len(l_invalid[0])):

            # Get pixel coordinates - O(1)
            y = l_invalid[0][i]
            x = l_invalid[1][i]

            # Get actual search window size and cut thd for the actual pixel - O(1)
            window_size = l_window_sizes[y][x]
            l_cut_thd = l_cut_thds[y][x]

            # Get block valid mask
            l_reciprocity_mask_block = get_block(l_reciprocity_mask_temp, x, y, window_size)

            # Get recovered block
            l_recovered_block = get_block(l_colorized, x, y, window_size)
            blue_block, green_block, red_block = cv.split(l_recovered_block)

            # Get similarity weights from original channel and masks it
            l_original = get_block(red_channel, x, y, window_size)
            l_weights = get_weights(l_original)

            # Discards dissimilar pixels
            l_cut_mask = np.where(l_weights > l_cut_thd, 0, 1)
            l_cut_mask = np.where(l_reciprocity_mask_block == 1, l_cut_mask, 0)

            # Masks blocks
            green_block_masked = np.where(l_cut_mask == 1, green_block, 0)
            blue_block_masked = np.where(l_cut_mask == 1, blue_block, 0)

            count = utils.count_if(l_cut_mask, 1)
            if count > min_matches:

                red = anaglyph[y][x][2]

                l_colorized[y][x][2] = red
                l_colorized[y][x][1] = int(round(np.sum(green_block_masked)/count))
                l_colorized[y][x][0] = int(round(np.sum(blue_block_masked)/count))

                l_reciprocity_mask_temp[y][x] = 1
                l_invalid_count -= 1
            else:
                if l_window_sizes[y][x] + window_increment < max_window_size:
                    l_window_sizes[y][x] = l_window_sizes[y][x] + window_increment
                else:
                    l_cut_thds[y][x] = l_cut_thds[y][x] + 1

        if SHOW_RESULTS:
            cv.imshow("Left Colorized", l_colorized)
            cv.imshow("Left Valid Mask", utils.convert_to_image(l_reciprocity_mask_temp))
            cv.waitKey(1)

    # While there are invalid pixels
    while r_invalid_count > 0:

        # Get invalid borders
        r_invalid = get_invalid_borders(r_reciprocity_mask_temp)

        # For each invalid pixel
        for i in range(len(r_invalid[0])):

            # Get pixel coordenates
            y = r_invalid[0][i]
            x = r_invalid[1][i]

            # Get actual search window size and cut thd for the actual pixel
            window_size = r_window_sizes[y][x]
            r_cut_thd = r_cut_thds[y][x]

            # Get block valid mask
            r_reciprocity_mask_block = get_block(r_reciprocity_mask_temp, x, y, window_size)

            # Get recovered block
            r_recovered_block = get_block(r_colorized, x, y, window_size)
            blue_block, green_block, red_block = cv.split(r_recovered_block)

            # Get similarity weights from original channel and masks it
            r_original = get_block(cyan_channel, x, y, window_size)
            r_weights = get_weights(r_original) #*r_reciprocity_mask_block

            # Discards dissimilar pixels
            r_cut_mask = np.where(r_weights > r_cut_thd, 0, 1)
            r_cut_mask = np.where(r_reciprocity_mask_block == 1, r_cut_mask, 0)

            # Masks blocks
            red_block_masked = r_cut_mask * red_block

            count = utils.count_if(r_cut_mask, 1)

            if count > min_matches:

                red = int(round(np.sum(red_block_masked)/count))
                green = anaglyph[y][x][1]
                blue = anaglyph[y][x][0]

                r_colorized[y][x][2] = red
                r_colorized[y][x][1] = green
                r_colorized[y][x][0] = blue

                r_reciprocity_mask_temp[y][x] = 1
                r_invalid_count -= 1
            else:
                if r_window_sizes[y][x] + window_increment < max_window_size:
                    r_window_sizes[y][x] = r_window_sizes[y][x] + window_increment
                else:
                    r_cut_thds[y][x] = r_cut_thds[y][x] + 1

        if SHOW_RESULTS:
            cv.imshow("Right Colorized", r_colorized)
            cv.imshow("Right Valid Mask", utils.convert_to_image(r_reciprocity_mask_temp))
            cv.waitKey(1)

    return l_colorized, r_colorized
