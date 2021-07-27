SHOW_RESULTS = False
SHOW_OFF_MODE = False

PATH = "./database"
L_PATH = PATH + "/dataset/left_side/"
R_PATH = PATH + "/dataset/right_side/"

ANAGLYPH_CONFIG = {
    'color_scheme': 'rc',
    'right_left_factor': 1,
}

REPRESENTATION_CONFIG = {
    'mode': 'DifferentialCrop',
    'window_size': 13,
}

AGGREGATION_CONFIG = {
    # window size = 2*int(truncate*sigma + 0.5) + 1
    'sigma': 4,
    'truncate': 3,
}

RECIPROCITY_CONFIG = {
    'threshold': 1,
}

REFINEMENT_CONFIG = {
    'k_size': 35,
}

COLORIZATION_CONFIG = {
    'min_matches': 2,
    'threshold': 5,
    'min_window_size': 43,
    'window_increment': 12,
    'erosion_kernel': 3
}

IMAGES = [
    {'name':'Tsukuba', 'min_disp':3, 'max_disp':15, 'scale_factor':1},
    {'name':'Venus', 'min_disp':3, 'max_disp':20, 'scale_factor':8},
    {'name':'Cones', 'min_disp':1, 'max_disp':55, 'scale_factor':4},
    {'name':'Teddy', 'min_disp':12, 'max_disp':53, 'scale_factor':4},
    {'name':'Sawtooth', 'min_disp':3, 'max_disp':18, 'scale_factor':8},
    {'name':'Bull', 'min_disp':3, 'max_disp':20, 'scale_factor':8},
    {'name':'Poster', 'min_disp':3, 'max_disp':21, 'scale_factor':8},
    {'name':'BarnOne', 'min_disp':3, 'max_disp':17, 'scale_factor':8},
    {'name':'BarnTwo', 'min_disp':3, 'max_disp':17, 'scale_factor':8},
    {'name':'Art', 'min_disp':11, 'max_disp':104, 'scale_factor':2.171875},
    {'name':'Books', 'min_disp':3, 'max_disp':102, 'scale_factor':2.171875},
    {'name':'Dolls', 'min_disp':4, 'max_disp':102, 'scale_factor':2.171875},
    {'name':'Laundry', 'min_disp':8, 'max_disp':111, 'scale_factor':2.096875},
    {'name':'Moebius', 'min_disp':19, 'max_disp':101, 'scale_factor':2.171875},
    {'name':'Reindeer', 'min_disp':5, 'max_disp':96, 'scale_factor':2.096875},
    {'name':'Computer', 'min_disp':11, 'max_disp':81, 'scale_factor':2.078125},
    {'name':'Drumsticks', 'min_disp':13, 'max_disp':100, 'scale_factor':2.171875},
    {'name':'Dwarves', 'min_disp':20, 'max_disp':90, 'scale_factor':2.171875},
]
