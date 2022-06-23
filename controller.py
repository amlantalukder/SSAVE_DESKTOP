import os, sys, re
import numpy as np
import pdb

import visualize_sleep as scv

# --------------------------------------------------------------------------
class Controller():
    
    # --------------------------------------------------------------------------
    def removeOutputFiles(self):

        img_path = os.path.join(self.scv_obj.folder_cache, f'{self.scv_obj.sample_name}.jpg')
        sc_path = os.path.join(self.scv_obj.folder_cache, f'{self.scv_obj.sample_name}_sc.txt')
        st_path = os.path.join(self.scv_obj.folder_cache, f'{self.scv_obj.sample_name}_st.txt')
        eeg_path = os.path.join(self.scv_obj.folder_cache, f'{self.scv_obj.sample_name}_eeg.npy')
        spec_path = os.path.join(self.scv_obj.folder_cache, f'{self.scv_obj.sample_name}_spec_2.npy')

        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(sc_path): os.remove(sc_path)
        if os.path.exists(st_path): os.remove(st_path)
        if os.path.exists(eeg_path): os.remove(eeg_path)
        if os.path.exists(spec_path): os.remove(spec_path)

    # --------------------------------------------------------------------------
    def validate(self, value, _type, input_name=''):

        if _type == 'file_type':
            if value in ['edf', 'annot']: return value
            print("Error!! sample type not selected")
        elif _type == 'epoch_size': return float(value)
        elif _type == 'input_file_path':
            if os.path.exists(value) and os.path.isfile(value) and os.path.splitext(value)[1].lower() in ['.edf', '.txt']: return os.path.abspath(value)
            print("Error!! invalid sample path selected")
        elif _type == 'output_folder_path':
            if os.path.exists(value) and os.path.isdir(value): return os.path.abspath(value)
            print("Error!! invalid output path selected")
        elif _type in ['frequency', 'amplitude', 'time']:
            try:
                return float(value)
            except:
                print("Error!! invalid value for {input_name}")

# --------------------------------------------------------------------------
class DesktopUIController(Controller):

    channels_all = None
    annotations_all = None
    state_changed = True
    
    # --------------------------------------------------------------------------
    def loadSleepData(self, view):

        file_type = self.validate(view.file_type_entry.get(), 'file_type')
        input_file_path = self.validate(view.sample_path_entry.get(), 'input_file_path')
        output_folder_path = self.validate(view.output_path_entry.get(), 'output_folder_path')

        if (file_type is None) or (input_file_path is None) or (output_folder_path is None): return

        print(input_file_path, file_type, output_folder_path)

        try:
            self.scv_obj = scv.loadSleepData(input_file_path, output_folder_path, file_type)
            if file_type == 'edf':
                if self.scv_obj.eeg_data is None: self.scv_obj.loadEEG()
                self.annotations_all = np.sort(np.unique(self.scv_obj.eeg_data._annotations.description))
                self.channels_all = self.scv_obj.eeg_data.info['ch_names']
                if len(self.channels_all) == 0: print('No channels found !!!')
                scv.Config.CHANNELS_SELECTED = scv.Config.CHANNELS_SELECTED[np.in1d(scv.Config.CHANNELS_SELECTED, self.channels_all)]
        except:
            return False

        print('Loaded sleep data successfully ...')

        self.state_changed = True

        return True

    # --------------------------------------------------------------------------
    def execute(self, view):

        if self.scv_obj.apply_filter != view.apply_filter.get():
            self.scv_obj.apply_filter = view.apply_filter.get()
            self.state_changed = True
        
        if view.cut_table and self.scv_obj.cut_options_selected != view.cut_table.get():
            self.scv_obj.cut_options_selected = view.cut_table.get()
            self.state_changed = True

        if self.state_changed:
            self.removeOutputFiles()
            self.scv_obj.clearSleepData()
        
            try:
                scv.extractSleepStages(self.scv_obj, self.scv_obj.apply_filter)
            except scv.SpecialError as error:
                print(error.msg)

            self.state_changed = False

    # --------------------------------------------------------------------------
    def getConfig(self):
        return scv.Config

    # --------------------------------------------------------------------------
    def saveSettings(self, view):

        channels_selected = np.array([ch_name for ch_name, var in view.channel_values.items() if var.get() == True])

        for st_stage_ind, stage in enumerate(scv.Config.SLEEP_STAGES_ALL):
            no_st_selection = True
            for annot in view.annot_checkbuttons_right[st_stage_ind]:
                if view.annot_checkbuttons_right[st_stage_ind][annot]:
                    no_st_selection = False
                    scv.Config.sleep_stage_event_to_id_mapping[annot.lower()] = stage

            if no_st_selection: return (False, f"No annotation selected for {scv.Config.SLEEP_STAGE_ALL_NAMES[st_stage_ind]}")

        notch_freq = self.validate(view.notch_freq_entry.get(), 'frequency', input_name='Notch Frequency')
        bandpass_min_freq = self.validate(view.bandpass_min_freq_entry.get(), 'frequency', input_name='Bandpass Maximum Frequency')
        bandpass_max_freq = self.validate(view.bandpass_max_freq_entry.get(), 'frequency', input_name='Bandpass Minimum Frequency')
        amplitude_max = self.validate(view.amplitude_max_entry.get(), 'amplitude', input_name='Maximum Amplitude')
        flat_signal_duration = self.validate(view.flat_signal_duration_entry.get(), 'time', input_name='Flat Signal Duration')
        bad_annots = [annot for annot, var in view.bad_annots_sel.items() if var.get() == True]
        epoch_size = self.validate(view.epoch_size_entry.get(), 'time', input_name='Epoch Size')

        if len(channels_selected) == 0:
            return (False, "No channels selected")

        if (notch_freq is None) or (bandpass_min_freq is None) or (bandpass_max_freq is None) or (amplitude_max is None) or (flat_signal_duration is None):
            return (False, "Invalid filter input")

        if bandpass_max_freq < bandpass_min_freq:
            return (False, "Bandpass maximum frequency cannot be less than minimum frequency")

        if (epoch_size is None):
            return (False, "Invalid epoch size")

        scv.Config.FILTERS['notch'] = notch_freq
        scv.Config.FILTERS['bandpass'] = [bandpass_min_freq, bandpass_max_freq]
        scv.Config.FILTERS['amplitude_max'] = amplitude_max
        scv.Config.FILTERS['flat_signal'] = flat_signal_duration
        scv.Config.FILTERS['bad_annots'] = bad_annots
        scv.Config.CHANNELS_SELECTED = channels_selected
        scv.Config.EPOCH_SIZE = epoch_size

        #print(scv.Config.sleep_stage_event_to_id_mapping)

        self.state_changed = True

        return (True, "Saved Settings")

# --------------------------------------------------------------------------
class CLIController(Controller):

    # --------------------------------------------------------------------------
    def loadSleepData(self, view):

        input_file_path = self.validate(input('Enter Sample Path: '), 'input_file_path')
        while input_file_path is None:
            input_file_path = self.validate(input('Enter Sample Path: '), 'input_file_path')

        file_type = input('Select File Type: 1. EDF format 2. Annotation epochs: ') 
        while file_type not in ['1', '2']: 
            file_type = input('Select File Type: 1. EDF format 2. Annotation epochs: ') 
        file_type = 'edf' if file_type == '1' else 'annot'
        
        epoch_size =  self.validate(input('Enter Epoch Size: '), 'epoch_size')

        output_folder_path =  self.validate(input('Enter Output Folder Path: '), 'output_folder_path')
        while output_folder_path is None:
            output_folder_path =  self.validate(input('Enter Output Folder Path: '), 'output_folder_path')

        apply_filter =  input('Apply Filter: 1. Yes 2. No: ')
        while apply_filter not in ['1', '2']:
            apply_filter = input('Apply Filter: 1. Yes 2. No: ') 
        apply_filter = True if apply_filter == '1' else False

        if (file_type is None) or (input_file_path is None) or (epoch_size is None) or (output_folder_path is None): return

        print(input_file_path, file_type, epoch_size, output_folder_path)

        try:
            self.scv_obj = scv.loadSleepData(input_file_path, output_folder_path, file_type, epoch_size)
            self.scv_obj.apply_filter = apply_filter
            if file_type == 'edf':
                if self.scv_obj.eeg_data is None: self.scv_obj.loadEEG()
                self.annotations_all = np.sort(np.unique(self.scv_obj.eeg_data._annotations.description))
                self.channels_all = self.scv_obj.eeg_data.info['ch_names']
                if len(self.channels_all) == 0: print('No channels found !!!')
                scv.Config.CHANNELS_SELECTED = scv.Config.CHANNELS_SELECTED[np.in1d(scv.Config.CHANNELS_SELECTED, self.channels_all)]
        except:
            return False

        print('Loaded sleep data successfully ...')

        return True

    # --------------------------------------------------------------------------
    def execute(self, view):

        self.removeOutputFiles()
        self.scv_obj.clearSleepData()
        try:
            scv.extractSleepStages(self.scv_obj)
        except scv.SpecialError as error:
            print(error.msg)