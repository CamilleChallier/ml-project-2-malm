from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sleep_eeg.model_comparaison import *
from sleep_eeg.pre_processing.sleep_cycles import *
from sleep_eeg.plotting.sleep_cycles import *
from sleep_eeg.plotting.morning_questionnaire import *
from sleep_eeg.plotting.model_comparaison import *
from sleep_eeg.utils import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sleep_eeg.utils import *
from sleep_eeg.pre_processing.eeg_preprocessing import *
from sleep_eeg.pre_processing.eeg_features_generation import *

eeg_list =['EEG A1','EEG A2','EEG M2', 'EEG E1', "EEG E2", 'EEG M1', 'EEG Fp1', 'EEG Fp2','EEG F3', 'EEG F4', 'EEG C3', 'EEG C4', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG Fz', 'EEG Pz', 'EEG Cz', 'EEG Oz']
other_list = [ 'POS', 'FlowT', 'THO', 'TibR', 'ECG ECG', 'Snore', 'Chin1', 'TibL', 'Chin2', 'ABD', 'FlowP']

directory = "data\eeg"

commun_preprocessor_pipe = Pipeline_pre_processor(
    steps=[
        (Renamer, None),
        (Cropper, {"tmin": 60, "tmax": None}),
        (Resampler, {"sfreq": 100}),
        (Standard_scaler, {"channels": other_list }),
        ])

eeg_preprocessor_pipe = Pipeline_pre_processor(
                steps=[
                    (Mean_remover, None),
                    (Common_average_reference, None),
                    (Filter, {"low_cut": 0.5, "hi_cut": 49.9})  
                ])

features_generator_pipe = Pipeline_features_generator(
                     steps=[
                         (WaveletFeatureGenerator, {"sfreq": 100,
                                                    "wavelet": "db4",
                                                    "band_limits": [[0, 2], [2, 4],  [4, 8], [8, 13],[13, 18],  [18, 24], [24, 30], [30, 49.9]],
                                                    "domain" : "dwt"}),
                        (Fourrier, {"bands": [[0, 2], [2, 4],  [4, 8], [8, 13],[13, 18],  [18, 24], [24, 30], [30, 49.9]], 
                                    "sfreq": 100}),
                        (Sex_feature, None),
                        (TiB_features, None),
                        (Night_lenght, None),
                        (ECG_features, None),
                        (Time_features,None)
                    ],
                )
if __name__ == "__main__":
        
    df = pd.DataFrame()

    for subdir, _, files in os.walk(directory):
        for file in files:
            print(file)
            filepath = subdir + os.sep + file
            if filepath.endswith(".edf"):
                
                # if not (file.split(".")[0] in col):
            
                raw = load_eeg(filepath)
                
                if len(raw.info["subject_info"]) >= 4  and len(raw.info["ch_names"]) >= 37:
                
                    filterd_raw = commun_preprocessor_pipe.fit_transform(raw)
                    del raw
                    raw_eeg = filterd_raw.copy().pick_channels(eeg_list)
                    filtered_eeg = eeg_preprocessor_pipe.fit_transform(raw_eeg)
                    del raw_eeg
                    final_raw = filtered_eeg.add_channels([filterd_raw.pick(other_list)], force_update_info=True)

                    features_df = features_generator_pipe.generate_features(final_raw)
                    features_df["NAME"] = file.split(".")[0]
                    df = pd.concat([df,features_df], axis=0) 
                    
                    if len(df)%5 ==0 : 
                        df.to_csv("../data/eeg/df_corrected_{}.csv".format(len(df)))
                    del features_df
                    del filterd_raw
                    del filtered_eeg
                    del final_raw
                        
    df.to_csv("../data/eeg/df_{}.csv".format(len(df)))
        