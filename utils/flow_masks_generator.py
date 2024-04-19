from pathlib import Path
import pandas as pd
import numpy as np

database_dirpath = Path('../../../CASME3/processed_data/')

annotations_path = Path('../../../../../media/harshamupparaju/Expansion/Harsha/Databases/CASME3/part_A/annotation/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx')

annotations = pd.read_excel(annotations_path)

for index, row in annotations.iterrows():
    subject = row['Subject']
    filename = row['Filename']
    onset = row['Onset']
    apex = row['Apex']
    offset = row['Offset']

    datapoint_dirpath = database_dirpath / f'{subject}_{filename}_{onset}_{offset}'
    rgb_dirpath = datapoint_dirpath / 'rgb'

    frames = [frame.stem for frame in rgb_dirpath.glob('*.jpg')]

    onset_frame = [frame for frame in frames if frame.split('_')[-1] == str(onset)][0]

    apex_frame = 
    