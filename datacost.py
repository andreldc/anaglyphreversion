import numpy as np
import cv2 as cv
import utils
from config import SHOW_RESULTS, SHOW_OFF_MODE

WAITKEY_TIME = 1
if SHOW_OFF_MODE:
    WAITKEY_TIME = 0

def get_data_costs(l_channel, r_channel, l_representations, r_representations, min_disp, max_disp):

    y_axis, x_axis, _ = l_representations.shape

    # Variable to store Data Costs - O(1)
    l_costs = np.ones((max_disp + 1, y_axis, x_axis))*float('inf')
    r_costs = np.ones((max_disp + 1, y_axis, x_axis))*float('inf')

    # For each disparity level - (L.N.w²)
    for disp in range(min_disp, max_disp + 1):
        utils.progress_bar(disp - min_disp, max_disp - min_disp, text="Data Cost Computation")

        # Crop representations - O(1)
        l_representations_crop = l_representations[:, disp:, :]
        r_representations_crop = r_representations[:, :-disp, :]

        # Get data Cost - O(N.w²)
        costs = utils.SAD(l_representations_crop, r_representations_crop)

        #Pads representation arrays back to original size - O(1)
        l_r_costs = np.pad(costs, ((0, 0), (abs(disp), 0)), 'constant',
                           constant_values=(costs.max(), 0))

        r_l_costs = np.pad(costs, ((0, 0), (0, abs(disp))), 'constant',
                           constant_values=(0, costs.max()))

        # Saves data cost - O(1)
        l_costs[disp] = l_r_costs
        r_costs[disp] = r_l_costs

        if SHOW_RESULTS:
            # Pads Sliding Window
            l_channel_pad = \
                np.pad(l_channel, ((0, 0), (-disp if disp < 0 else 0, disp if disp >= 0 else 0)),
                       'constant', constant_values=(0, 0))

            r_channel_pad = \
                np.pad(r_channel, ((0, 0), (disp if disp >= 0 else 0, -disp if disp < 0 else 0)),
                       'constant', constant_values=(0, 0))

            # Shows Images
            cv.imshow("Sliding Window",
                      utils.stack_1_by_2(l_channel_pad, r_channel_pad, orientation="Vertical"))

            utils.show_image_stack("Data Costs", utils.stack_1_by_2(l_costs, r_costs),
                                   min_val=1, max_val=max_disp, initial_pos=disp)

            cv.waitKey(1)

    if SHOW_RESULTS:
        cv.imshow("Costs (Click for details)", l_channel)
        cv.setMouseCallback("Costs (Click for details)", utils.preview_array,
                            param=(l_costs, r_costs, "SAD", "Disparidade"))
        cv.waitKey(WAITKEY_TIME)

    return l_costs, r_costs
