import sys
import time
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def change_range(vector, old_min, old_max, new_min, new_max):
    """
    Scale np.array from range old_min, old_max be within new_min and new_max
    """

    if(vector.min() != vector.max() and old_max != old_min):
        new_vector = ((vector-old_min)/(old_max-old_min))*(new_max-new_min) + new_min
    else:
        new_vector = vector

    return new_vector


def split_block_channels(block):
    """
    Splits a given block in its flatten channels
    """

    l_representations = block[:, :, 0].flatten()
    r_representations = block[:, :, 1].flatten()

    return l_representations, r_representations


def normalize(vector, new_min=0, new_max=255):
    """
    Scale vector to be within min_val and max_val relative to its own min and max values
    """

    old_min = vector.min()
    old_max = vector.max()

    return change_range(vector, old_min, old_max, new_min, new_max)


def get_gaussian_block(block_size, sigma=-1, percent=False):
    """
    Returns Gaussian Block
    """

    gauss = np.dot(cv.getGaussianKernel(block_size, sigma),
                   cv.getGaussianKernel(block_size, sigma).transpose())

    if percent:
        gauss = gauss/gauss.max()

    return gauss


class MyTimer:
    """
    Timer class
    """

    def __init__(self):
        self.start = time.time()
        self.last_time = time.time()

    def get_actual_time(self, identifier):
        """
        Returns actual timer and prints to screen
        """

        actual_time = time.time()
        partial = actual_time - self.last_time
        total = actual_time - self.start

        self.last_time = actual_time

        minutes_partial = int(partial/60)
        minutes_total = int(total/60)

        seconds_partial = partial - 60*minutes_partial
        seconds_total = total - 60*minutes_total

        text_partial = ""
        if minutes_partial > 0:
            text_partial = "{:05.2f}m {:05.2f}s".format(minutes_partial, seconds_partial)
        else:
            text_partial = "{:05.2f}s".format(seconds_partial)

        text_total = ""
        if minutes_total > 0:
            text_total = "{:05.2f}m {:05.2f}s".format(minutes_total, seconds_total)
        else:
            text_total = "{:05.2f}s".format(seconds_total)


        cprint(identifier, "{} / {}".format(text_partial, text_total), Cl.OKGREEN)

        return partial, total



def convert_to_image(vector):
    """
    Converts vector to image (0-255 uint)
    """
    return np.rint(normalize(vector, 0, 255)).astype("uint8")


def SAD(reps1, reps2, axis=2):
    """
    Returns the Sum of Absolute Differences
    """
    return np.sum(np.absolute(reps1.astype('float') - reps2.astype('float')), axis=axis)


def show_image_stack(name, image_stack, min_val=0, max_val=None, initial_pos=0):
    """
    Shows image Stack
    """

    if max_val is None:
        max_val = image_stack.shape[0]

    title_window = name
    trackbar_name = name + '%d' % image_stack.shape[0]

    cv.namedWindow(title_window)
    cv.createTrackbar(trackbar_name, title_window, min_val, max_val, \
                      lambda val: cv.imshow(title_window, convert_to_image(image_stack[val])))
    cv.setTrackbarPos(trackbar_name, title_window, initial_pos)


def show_block(block, name="block", size=300, center_color=None):
    """
    Shows block
    """
    shape = block.shape

    if len(shape) == 2:
        block = dstack_3(block)

    if center_color is not None:
        hws = int(block.shape[0]/2)
        block[hws, hws] = center_color

    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, size, size)
    cv.imshow(name, convert_to_image(block))


def dstack_3(arrays):
    """
    Merge 3 arrayss
    """
    return np.dstack((arrays, arrays, arrays))


def mono_colored(image, color):
    """
    Turns monocromatic image to colored mono
    """
    zeroes = np.zeros(image.shape).astype('uint8')
    # print(image.shape, zeroes.shape)

    if color == 'red':
        out = cv.merge((zeroes, zeroes, image))
    elif color == 'green':
        out = cv.merge((zeroes, image, zeroes))
    elif color == 'blue':
        out = cv.merge((image, zeroes, zeroes))

    return out


def stack_1_by_2(image_1, image_2, convert_image=False, orientation="Horizontal"):
    """
    Stacks 2 np.arrays vertically or horizontally
    """
    if convert_image:
        image_1 = convert_to_image(image_1)
        image_2 = convert_to_image(image_2)

    if orientation == "Horizontal":
        out = np.hstack((image_1, image_2))
    elif orientation == "Vertical":
        out = np.vstack((image_1, image_2))
    else:
        out = np.hstack((image_1, image_2))

    return out

def stack_1_by_3(image_1, image_2, image_3, convert_image=False, orientation="Horizontal"):
    """
    Stacks 2 np.arrays vertically or horizontally
    """
    if convert_image:
        image_1 = convert_to_image(image_1)
        image_2 = convert_to_image(image_2)
        image_3 = convert_to_image(image_3)

    if orientation == "Vertical":
        out = np.vstack((image_1, image_2, image_3))
    else:
        out = np.hstack((image_1, image_2, image_3))

    return out


def stack_2_by_2(image_1, image_2, image_3, image_4, convert_image=False):
    """
    Stacks 4 np.arrays
    """
    if convert_image:
        image_1 = convert_to_image(image_1)
        image_2 = convert_to_image(image_2)
        image_3 = convert_to_image(image_3)
        image_4 = convert_to_image(image_4)

    row1 = np.hstack((image_1, image_2))
    row2 = np.hstack((image_3, image_4))

    return np.vstack((row1, row2))


def progress_bar(value, end_value, bar_length=27, text=""):
    """
    Shows/updates progress bar
    """

    percent = float(value) / end_value
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r[{1}] {2}% - {0}".format(text, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

    if round(percent*100) >= 99:
        sys.stdout.write("\r" + " " * 100 + "\r")


def count_if(array, value):
    """
    Count number of "value" are in the array
    """
    unique, counts = np.unique(array, return_counts=True)
    totals = dict(zip(unique, counts))

    if value in totals:
        total = totals[value]
    else:
        total = 0

    return total


class Cl:
    """
    Text Colors
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cprint(title, data="", title_color=Cl.CBEIGE, data_color=Cl.CWHITE):
    """
    Prints formatted text
    """
    print(title_color + str(title).ljust(30) + Cl.ENDC + data_color + str(data) + Cl.ENDC)


def dict_to_csv(dictionary):
    """
    Converts dictionary to CSV
    """

    headers = ""
    values = ""

    for key in dictionary.keys():
        headers += str(key) + ";"
        values += str(dictionary[key]) + ";"

    return headers, values


def time_dict_to_csv(dictionary):
    """
    Converts dictionary to CSV
    """

    time_headers = ""
    time_values = ""

    for key in dictionary.keys():
        time_headers = time_headers + "t_" + key + ";"
        time_values = time_values +  "{:03.4f}".format(dictionary[key][0]) + ";"

    return time_headers, time_values


def check_and_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def label_image(image, label, init_x=18, init_y=18, color=[100, 255, 0]):
    """
    Add label to a given image
    """

    l_image = np.copy(image)
    end_y = init_y + 17
    end_x = init_x + len(label)*8+8

    cv.rectangle(l_image, (init_x, init_y), (end_x, end_y), [0, 0, 0], -1)
    cv.putText(l_image, label.lower(), (init_x + 2, init_y + 12), cv.FONT_HERSHEY_SIMPLEX, .5,
               color)
    return l_image


def get_block(image, x, y, window_size):
    ''' Get a block from a given image at coordinates x,y '''
    half_window_size = int(window_size/2)

    bordered = cv.copyMakeBorder(image, half_window_size, half_window_size,
                                 half_window_size, half_window_size, cv.BORDER_REFLECT_101)

    return bordered[y:y + window_size, x:x + window_size]


def get_image_texture(image, window_size, threshold=1):
    """
    Get image textures and mask
    """
    half_window_size = int((window_size-1)/2)
    y_axis, x_axis, _ = image.shape

    textures = np.empty([y_axis, x_axis])

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    bordered = cv.copyMakeBorder(gray, half_window_size, half_window_size,
                                 half_window_size, half_window_size,
                                 cv.BORDER_REFLECT_101)

    # --- For each pixel in the image that fits within the block
    for y in range(0, y_axis):
        for x in range(0, x_axis):
            textures[y, x] = int(np.std(bordered[y:y + window_size, x:x + window_size]))

    mask = np.where(textures < threshold, 0, 1).astype('uint8')

    return textures, mask

def get_radiometric_differences(l_image, r_image, l_disparity, r_disparity):

    y_axis, x_axis, _ = l_image.shape

    l_blue, l_green, l_red = cv.split(l_image)
    r_blue, r_green, r_red = cv.split(r_image)

    l_gray = cv.cvtColor(l_image, cv.COLOR_BGR2GRAY).astype('int16')
    r_gray = cv.cvtColor(r_image, cv.COLOR_BGR2GRAY).astype('int16')

    cyan = np.rint(.75*r_green + .25*r_blue).astype('uint8')

    cv.imshow("channels", stack_1_by_2(l_red, stack_1_by_2(r_green, r_blue)))
    cv.imshow("channels2", stack_1_by_2(l_red, cyan))
    cv.waitKey(1)

    cv.imwrite("C:/Users/andre/OneDrive/Pessoais/Escolares/07 - Mestrado USP/06 - Defesa/images/databaseEval/l_radiometric_example.jpg", convert_to_image(l_red))
    cv.imwrite("C:/Users/andre/OneDrive/Pessoais/Escolares/07 - Mestrado USP/06 - Defesa/images/databaseEval/r_radiometric_example.jpg", convert_to_image(cyan))

    l_red = l_red.astype('int16')
    r_green = r_green.astype('int16')
    r_blue = r_blue.astype('int16')

    l_diffs_r_g = np.zeros((y_axis, x_axis))
    l_diffs_r_b = np.zeros((y_axis, x_axis))

    r_diffs_r_g = np.zeros((y_axis, x_axis))
    r_diffs_r_b = np.zeros((y_axis, x_axis))

    # --- For each pixel in the image that fits within the block
    for y in range(0, y_axis):
        for x in range(0, x_axis):

            l_disp = l_disparity[y][x]

            if l_disp > 0:
                l_pixel = l_red[y][x]
                r_pixel1 = r_green[y][x - l_disp]
                r_pixel2 = r_blue[y][x - l_disp]

                l_diffs_r_g[y][x] = abs(r_pixel1 - l_pixel)
                l_diffs_r_b[y][x] = abs(r_pixel2 - l_pixel)

            r_disp = r_disparity[y][x]

            if r_disp > 0:
                l_pixel = l_red[y][x + r_disp]
                r_pixel1 = r_green[y][x]
                r_pixel2 = r_blue[y][x]

                r_diffs_r_g[y][x] = abs(r_pixel1 - l_pixel)
                r_diffs_r_b[y][x] = abs(r_pixel2 - l_pixel)

    l_diffs_r_g_mask = np.where(l_diffs_r_g > 64, 0, 1)
    l_diffs_r_b_mask = np.where(l_diffs_r_b > 64, 0, 1)
    r_diffs_r_g_mask = np.where(r_diffs_r_g > 64, 0, 1)
    r_diffs_r_b_mask = np.where(r_diffs_r_b > 64, 0, 1)

    return l_diffs_r_g, r_diffs_r_g, l_diffs_r_g_mask, r_diffs_r_g_mask

def preview_array(event, x, y, flags, param):

    if event == cv.EVENT_LBUTTONDOWN:

        left, right, ylabel, xlabel = param

        vector = left[ y, x]

        print("Coordinates: ", x, y)
        print("Vector: ", vector)
        print("Argmin: ", np.argmin(vector) +"\n")

        plt.clf()
        plt.plot(vector)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()
