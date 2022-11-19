import sys, os, pdb
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from itertools import islice
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

    CODE_VERSION = 'v2'

    # --------------------------------------------------------------------------
    # Folder path related constants
    # --------------------------------------------------------------------------
    FOLDER_HOME = '/ddn/gs1/home/talukdera2/eeg_dl'
    #FOLDER_HOME = '/Users/talukdera2/Desktop/projects/eeg_dl'
    FOLDER_GROUP = '/ddn/gs1/group/li3/DataFromLLI/SleepData_UNC/UNC_data/All_EDFs'
    FOLDER_RAW_DATA = f'{FOLDER_HOME}/data/raw_data/{CODE_VERSION}'
    FOLDER_PROCESSED_DATA = f'{FOLDER_HOME}/data/processed_data/{CODE_VERSION}'
    FOLDER_RESULTS = f'{FOLDER_HOME}/results/{CODE_VERSION}'
    FOLDER_SLEEP_DATA = f'{FOLDER_RAW_DATA}/edfs'
    #FOLDER_SLEEP_ANNOT_RAW = f'{FOLDER_RAW_DATA}/SleepData_UNC/UNC_data/Healthy_Controls/TXT_EDF'
    #FOLDER_SLEEP_ANNOT_RAW = f'{FOLDER_RAW_DATA}/SleepData_UNC/UNC_data/Healthy_Controls/TXT_EDF'
    FOLDER_SLEEP_ANNOT_RAW = f'{FOLDER_RAW_DATA}/annots'
    FOLDER_SLEEP_ANNOT = f'{FOLDER_PROCESSED_DATA}/annots'

    # --------------------------------------------------------------------------
    # Plot related constants
    # --------------------------------------------------------------------------
    PLOT_DPI = 150
    PLOT_TITLE_FONT_SIZE = 25
    PLOT_LABEL_FONT_SIZE = 20
    PLOT_LEGEND_FONT_SIZE = 20
    PLOT_TICKLABEL_FONT_SIZE = 15

    # --------------------------------------------------------------------------
    # Common constants
    # --------------------------------------------------------------------------
    RANDOM_STATE = 1000

    # --------------------------------------------------------------------------
    # File chunk size to read at a time
    # --------------------------------------------------------------------------
    FILE_READ_CHUNK_SIZE = 1000_000_000
    
    # --------------------------------------------------------------------------
    # Training related constants
    # --------------------------------------------------------------------------
    TRAIN_MODEL = 1
    LOAD_MODEL = 2

    # --------------------------------------------------------------------------
    # Sample related constants
    # --------------------------------------------------------------------------
    CHANNELS_SELECTED = np.array(['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'M1', 'M2'])
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
    FILTERS = {'notch': 60.,  # [Hz]
        'bandpass': [0.05, 30.],  # [Hz]
        'amplitude_max': 500,
        'flat_signal': 5,
        'bad_annots': []
    }

    # --------------------------------------------------------------------------
    # Data processing related constants
    # --------------------------------------------------------------------------
    #SAMPLING_FREQ = 'default'
    SAMPLING_FREQ = 256
    EPOCH_SIZE = 30
    CONSIDER_DURATION_BETWEEN_LIGHTS_ON_OFF = True
    W_SIZE_DATA_STREAM, W_SLIDE_DATA_STREAM = 30, 30
    W_SIZE_EPOCH_STREAM, W_SLIDE_EPOCH_STREAM = 10, 10
    NUM_SAMPLES_PER_FILE = 10
    AMP_MICROVOLT_THRESHOLD_HIGH = 500
    FREQ_THRESHOLD_HIGH = 0.5
    FREQ_THRESHOLD_LOW = 20

# --------------------------------------------------------------------------
def loadFileNames(edf_source, ext):
    return sorted([item for item in os.listdir(edf_source)
                    if (os.path.isfile(os.path.join(edf_source, item)) and 
                        item.split('.')[-1].lower() == ext.lower())])

# -----------------------------------------------
def convertIndexToBinaryLabels(labels):
        
    num_rows, num_cols = labels.shape[0], np.max(labels)
    labels_bin = np.zeros((num_rows, num_cols))
    labels_bin[np.arange(labels.shape[0]), labels] = 1

    return labels_bin

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

# -----------------------------------------------------------------------------
def readBigFileInChunkGen(file_path, chunk_size=Config.FILE_READ_CHUNK_SIZE, header=False):

    with open(file_path) as f:
        while True:
            file_chunk = list(islice(f, chunk_size))
            if not file_chunk: break
            if header:
                file_chunk = file_chunk[1:]
                header = False
            yield file_chunk

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

# -------------------------------------
def getStats(a):

    quantile_25 = np.percentile(a, 25, interpolation='midpoint')
    quantile_75 = np.percentile(a, 75, interpolation='midpoint')
    return np.min(a), np.max(a), np.mean(a), np.median(a), stats.mode(a)[0][0], np.std(a), quantile_25, quantile_75

# ------------------------------------------------------------------------
def normalize(a, range=(0, 1), axis=None):
    range_size = range[1] - range[0]

    a = np.array(a)
    max_val = np.max(a, axis=axis)
    min_val = np.min(a, axis=axis)

    return range_size * (a - min_val) / (max_val - min_val) + range[0]

# ------------------------------------------------------------------------
def standardize(a, range=(0, 1), axis=None):

    a = np.array(a)
    mean = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)

    return (a - mean) / std

# -------------------------------------------
def multiplot(data_container, multi_plot_legends, title=None, xlabel=None, xticklabels=None, ylabel=None, plot_type='s', fig_name=None):
    
    fig, ax = plt.subplots()
    
    ax.set_title(title, fontsize=Config.PLOT_TITLE_FONT_SIZE)

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

    for i in range(len(data_container)):
        if plot_type == 's':
            ax.scatter(range(len(data_container[i])), data_container[i], c=colors[i % len(colors)], alpha=0.5,
                        label=multi_plot_legends[i])
        elif plot_type == 'l':
            x, y = range(len(data_container[i])), data_container[i]
            ax.plot(x, y, c=colors[i % len(colors)], alpha=0.5, label=multi_plot_legends[i])
            #ax.fill_between(x, y, color=colors[i % 4], alpha=0.2)

    ax.legend(fontsize=Config.PLOT_LEGEND_FONT_SIZE+2)

    if xlabel: ax.set_xlabel(xlabel, fontsize=Config.PLOT_LABEL_FONT_SIZE)
    if ylabel: ax.set_ylabel(ylabel, fontsize=Config.PLOT_LABEL_FONT_SIZE)

    if xticklabels:
        ax.set_xticklabels(xticklabels)
        step = max(x)//4
        ax.set_xticks(range(0, max(x), step))

    ax.tick_params(axis='both', labelsize=Config.PLOT_TICKLABEL_FONT_SIZE)

    if not fig_name: plt.show()
    else: 
        plt.savefig(fig_name)
        plt.close()

# -------------------------------------------
def plotBoxPlot(data_container, title='', x_ticklabels=[], x_label='', y_label='', colors=[], scale='', show_outliers=True, fig_name=None):
    
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    if len(x_ticklabels) == 0:
        bplot = ax1.boxplot(data_container)
    else:
        bplot = ax1.boxplot(data_container, labels=x_ticklabels, showfliers=show_outliers) #, notch=True, patch_artist=True)

    if colors:
        for item in ['boxes', 'medians', 'fliers']:
            for patch, color in zip(bplot[item], colors):
                if item != 'fliers': patch.set(color=color, linewidth=1)
                else: patch.set(marker='o', color=color, markeredgecolor=color)
                #if item == 'boxes': patch.set(facecolor=color)
        for item in ['caps', 'whiskers']:
            for i in range(len(colors)):
                bplot[item][i*2].set(color=colors[i], linewidth=1)
                bplot[item][i*2+1].set(color=colors[i], linewidth=1)
    
    if scale == 'log': ax1.set_yscale('log')
    ax1.set_title(title, fontsize=Config.PLOT_TITLE_FONT_SIZE)
    ax1.set_xlabel(x_label, fontsize=Config.PLOT_LABEL_FONT_SIZE)
    ax1.set_ylabel(y_label, fontsize=Config.PLOT_LABEL_FONT_SIZE-4)
    ax1.tick_params(axis='y', labelsize=Config.PLOT_TICKLABEL_FONT_SIZE)
    ax1.tick_params(axis='x', labelsize=Config.PLOT_TICKLABEL_FONT_SIZE-8)
    ax1.set_xticklabels(x_ticklabels, rotation=60, ha='right', rotation_mode='anchor')
    #plt.subplots_adjust(left=0.1, right=0.5, top=0.95, bottom=0.3)
    
    fig1.tight_layout()

    if not fig_name: plt.show()
    else:
        createDir(fig_name, is_file_path=True) 
        plt.savefig(fig_name)
        plt.close()

# -------------------------------------------
def createDir(d, is_file_path=False):
    
    if is_file_path: d = os.path.dirname(d)

    if d != "":
        if not os.path.exists(d):
            os.makedirs(d)

# -------------------------------------
def getStats(a, axis=None):
    import scipy as sp
    quartile_25 = np.percentile(a, 25, interpolation='midpoint', axis=axis)
    quartile_75 = np.percentile(a, 75, interpolation='midpoint', axis=axis)
    return np.min(a, axis=axis), np.max(a, axis=axis), np.mean(a, axis=axis), np.median(a, axis=axis), sp.stats.mode(a, axis=axis), np.std(a, axis=axis), quartile_25, quartile_75

# -------------------------------------
def getCorrCoeff(data, axis=0, corr_type='pearson'):
    import scipy as sp
    if corr_type == 'spearman':
        corr, pval = sp.stats.spearmanr(data, axis=axis)
        if np.isscalar(corr):
            return corr
        return corr.tolist()
    else:
        return np.corrcoef(data, rowvar=axis).tolist()

# ------------------------------------
def getTTest(a, b, axis=0, alternative='two-sided', type='ind'):
    import scipy as sp
    if type == 'ind':
        return sp.stats.ttest_ind(a, b, axis=axis, alternative=alternative).pvalue
    elif type == 'paired':
        return sp.stats.ttest_rel(a, b, axis=axis, alternative=alternative).pvalue
    else:
        pdb.set_trace()

# --------------------------------------------------------------------------
def getGroupWiseCounts(arr):
    group, count = [], 0
    for i in range(len(arr)):
        count += 1
        if i+1 == len(arr) or arr[i] != arr[i+1]:
            group.append((arr[i], count))
            count = 0
    return group