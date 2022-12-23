import sys, os, pdb
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import font_manager, rcParams
font_dir = ['web_version/static/css/Source_Serif_Pro']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

matplotlib.use('Agg')
rcParams['font.family'] = 'serif'

#plt.rcParams["font.family"] = "serif"

# --------------------------------------------------------------------------
class Config:

    # --------------------------------------------------------------------------
    # Plot related constants
    # --------------------------------------------------------------------------
    PLOT_DPI = 150
    PLOT_TITLE_FONT_SIZE = 25
    PLOT_LABEL_FONT_SIZE = 20
    PLOT_LEGEND_FONT_SIZE = 20
    PLOT_TICKLABEL_FONT_SIZE = 15

    # --------------------------------------------------------------------------
    # Channel settings
    # --------------------------------------------------------------------------
    CHANNELS_SELECTED = np.array(['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'M1', 'M2'])

    # --------------------------------------------------------------------------
    # Sleep stage settings
    # --------------------------------------------------------------------------    
    WAKE_STAGE, REM_STAGE, N1_STAGE, N2_STAGE, N3_STAGE, U_STAGE = 'W', 'R', 'N1', 'N2', 'N3', 'U'
    sleep_stage_event_to_id_mapping = {'sleep stage 1': N1_STAGE, 
                                        'sleep stage 2': N2_STAGE, 
                                        'sleep stage 3': N3_STAGE,
                                        'sleep stage 4': N3_STAGE,
                                        'sleep stage n': U_STAGE, 
                                        'sleep stage n1': N1_STAGE, 
                                        'sleep stage n2': N2_STAGE, 
                                        'sleep stage n3': N3_STAGE, 
                                        'sleep stage r': REM_STAGE,
                                        'sleep stage w': WAKE_STAGE}#,
                                        #'sleep stage ?': U_STAGE}
    SLEEP_STAGE_ALL_NAMES = ['WAKE STAGE', 'REM STAGE', 'N1 STAGE', 'N2 STAGE', 'N3 STAGE']#, 'Unassigned STAGE']
    SLEEP_STAGES = [WAKE_STAGE, REM_STAGE, N1_STAGE, N2_STAGE, N3_STAGE]
    SLEEP_STAGE_ANNOTS = {WAKE_STAGE:5, REM_STAGE:4, N1_STAGE:3, N2_STAGE:2, N3_STAGE:1}
    SLEEP_STAGES_ALL = [WAKE_STAGE, REM_STAGE, N1_STAGE, N2_STAGE, N3_STAGE]#, U_STAGE]#, 'O', 'NA']
    SLEEP_STAGE_ANNOTS_ALL = {WAKE_STAGE:5, REM_STAGE:4, N1_STAGE:3, N2_STAGE:2, N3_STAGE:1}#, U_STAGE:0}#, 'O':6, 'NA':7}
    
    # --------------------------------------------------------------------------
    # Filter setttings
    # --------------------------------------------------------------------------
    FILTERS = {'notch': 60.,  # [Hz]
        'bandpass': [0.05, 30.],  # [Hz]
        'amplitude_max': 500,
        'flat_signal': 5,
        'bad_annots': []
    }

    # --------------------------------------------------------------------------
    # Epoch settings
    # --------------------------------------------------------------------------
    EPOCH_SIZE = 30

# -----------------------------------------------
def writeFile(filename, data, mode="w"):
    createDir(filename, is_file_path=True)

    fl = open(filename, mode)
    fl.write(data)
    fl.close()

# -----------------------------------------------
def readFile(filename):

    fl = open(filename, "r")
    data = fl.readlines()
    fl.close()

    return data

# -----------------------------------------------
def readFileInTable(filename, delim='\t'):

    data = [item.strip().split(delim) for item in readFile(filename)]
    return data

# -----------------------------------------------
def writeDataTableAsText(data, filename, mode="w"):
    text = formatDataTable(data, "\t", "\n")

    writeFile(filename, text, mode)

# -----------------------------------------------
def formatDataTable(data, col_sep="\t", row_sep="\n"):
    return row_sep.join([col_sep.join([str(item1) for item1 in item]) for item in data])

# ------------------------------------
def printDec(msg):
    
    horizontal_border = '_' * 50
    vertical_border = '|'

    l = len(horizontal_border)

    print(horizontal_border)
    print(' ' * l)
    msg_part = msg.strip()
    while len(msg_part) >= l - 4:
        print(vertical_border + ' ' + msg_part[:l - 4] + ' ' + vertical_border)
        msg_part = msg_part[l - 4:].strip()
    print(vertical_border + ' ' + msg_part + ' ' * (l - 3 - len(msg_part)) + vertical_border)
    print(horizontal_border)
    print("")

# -----------------------------------------------
def showPercBar(counter, size, perc, perc_inc=10):
    num_prints = 100 // perc_inc
    progress = int(counter * 10 / size) * 10
    if progress >= perc:
        sys.stdout.write('=' * (int((progress - perc) // num_prints) + 1))
        sys.stdout.flush()
        if progress >= 100:
            print('100%')
        perc = progress + perc_inc
    return perc

# -------------------------------------------
def createDir(d, is_file_path=False):
    
    if is_file_path: d = os.path.dirname(d)

    if d != "":
        if not os.path.exists(d):
            os.makedirs(d)

# --------------------------------------------------------------------------
def getGroupWiseCounts(arr):
    group, count = [], 0
    for i in range(len(arr)):
        count += 1
        if i+1 == len(arr) or arr[i] != arr[i+1]:
            group.append((arr[i], count))
            count = 0
    return group