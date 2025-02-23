from pathlib import Path
import pandas as pd
import shutil
import numpy as np
import os
from tqdm import tqdm


root_dirpath = Path('../../../CASME3/')
database_dirpath = root_dirpath / 'processed_data'
flow_masks_dirpath = root_dirpath / 'flow_masks'
flow_masks_dirpath.mkdir(parents=True, exist_ok=True)
annotations_path = Path('../../../CASME3/processed_annotations/me_annotations.csv')

tmp_dirpath = Path('/mnt/2tb-hdd/Harsha/MER/tmp')
onset_apex_images_dirpath = tmp_dirpath / 'onset_apex/images'
apex_offset_images_dirpath = tmp_dirpath / 'apex_offset/images'


annotations = pd.read_csv(annotations_path)

for index, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
    subject = row['subject']
    filename = row['filename']
    onset = row['onset']
    apex = row['apex']
    offset = row['offset']

    datapoint_dirpath = database_dirpath / f'{subject}_{filename}_{onset}_{offset}'
    rgb_dirpath = datapoint_dirpath / 'rgb'

    datapoint_flow_masks_dirpath = flow_masks_dirpath / f'{subject}_{filename}_{onset}_{offset}'
    datapoint_flow_masks_dirpath.mkdir(parents=True, exist_ok=True)

    onset_apex_flow_mask_filepath = datapoint_flow_masks_dirpath / 'onset_apex.jpg'
    apex_offset_flow_mask_filepath = datapoint_flow_masks_dirpath / 'apex_offset.jpg'


    frames = [frame.stem for frame in rgb_dirpath.glob('*.jpg')]
    onset_frame = [frame for frame in frames if frame.split('_')[-1] == str(onset)][0]
    apex_frame = [frame for frame in frames if frame.split('_')[-1] == str(apex)][0]
    offset_frame = [frame for frame in frames if frame.split('_')[-1] == str(offset)][0]
  
    onset_apex_images_dirpath.mkdir(parents=True, exist_ok=True)
    apex_offset_images_dirpath.mkdir(parents=True, exist_ok=True)

    shutil.copy(rgb_dirpath / f'{onset_frame}.jpg', onset_apex_images_dirpath / f'{onset_frame}.jpg')
    shutil.copy(rgb_dirpath / f'{apex_frame}.jpg', onset_apex_images_dirpath / f'{apex_frame}.jpg')

    shutil.copy(rgb_dirpath / f'{apex_frame}.jpg', apex_offset_images_dirpath / f'{apex_frame}.jpg')
    shutil.copy(rgb_dirpath / f'{offset_frame}.jpg', apex_offset_images_dirpath / f'{offset_frame}.jpg')

    os.system(f'cd ../RAFT/ && python demo.py --model=models/raft-sintel.pth --path={onset_apex_images_dirpath} --output={onset_apex_flow_mask_filepath}')
    os.system(f'cd ../RAFT/ && python demo.py --model=models/raft-sintel.pth --path={apex_offset_images_dirpath} --output={apex_offset_flow_mask_filepath}')

    shutil.rmtree(onset_apex_images_dirpath)
    shutil.rmtree(apex_offset_images_dirpath)


    
    