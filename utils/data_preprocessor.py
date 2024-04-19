from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import shutil
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

from face_alignment import face_aligner
from face_detect_crop import face_cropper


def get_frames(rgb_dirpath : Path, depth_dirpath : Path, onset : int, apex : int, offset : int, frames_required : int) -> list:
    frames = []
    # print(onset, apex, offset)
    #Check if onset frame is present in present in rgb and depth
    if(offset < onset):
        offset = apex + 1
    while(len(frames) < 1):
        if(rgb_dirpath / f'{onset}.jpg').exists() and (depth_dirpath / f'{onset}.png').exists():
            frames.append(onset)
        else:
            onset += 1
    
    #Check if offset frame is present in present in rgb and depth
    while(len(frames) < 2):
        if(rgb_dirpath / f'{offset}.jpg').exists() and (depth_dirpath / f'{offset}.png').exists():
            frames.append(offset)
        else:
            offset -= 1
    
    while(len(frames) < 3):
        if( apex - onset <= offset - apex):
            update = 1
        else:
            update = -1
        if(rgb_dirpath / f'{apex}.jpg').exists() and (depth_dirpath / f'{apex}.png').exists():
            frames.append(apex)
        else:
            apex += update
    # print(onset, apex, offset)
    #Calculate the remaining frames
    remaining_frames = [i for i in range(onset + 1, offset) if (rgb_dirpath / f'{i}.jpg').exists() and (depth_dirpath / f'{i}.png').exists()]
    if(apex in remaining_frames):
        remaining_frames.remove(apex)

    if(len(remaining_frames) > frames_required - 3):
        remaining_frames = np.random.choice(remaining_frames, frames_required - 3, replace=False)
    
    for frame in remaining_frames:
        frames.append(frame)
    while(len(frames) < frames_required):
        frames.append(apex)

    frames.sort()

    return frames, onset, apex, offset

def pad_frames(rgb_frames : list, depth_frames : list):
    max_h = max([frame.shape[0] for frame in rgb_frames])
    max_w = max([frame.shape[1] for frame in rgb_frames])

    rgb_frames_padded = []
    depth_frames_padded = []

    for i in range(len(rgb_frames)):
        h, w = rgb_frames[i].shape[:2]
        pad_h = max_h - h
        pad_w = max_w - w

        rgb_frames_padded.append(np.pad(rgb_frames[i], ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=0))
        depth_frames_padded.append(np.pad(depth_frames[i], ((0, pad_h), (0, pad_w)), 'constant', constant_values=0))

    return np.array(rgb_frames_padded), np.array(depth_frames_padded)

database_dirpath = Path('../../../../../media/harshamupparaju/Expansion/Harsha/Databases/CASME3/part_A')

ME_output_dirpath = Path('../../../../2tb-hdd/CASME3/processed_data/')
ME_output_dirpath.mkdir(parents=True, exist_ok=True)

data_dirpath = database_dirpath / 'data/Compressed_version1_seperate_compress/'

me_annotation_path = database_dirpath / 'annotation/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx'
me_annotation = pd.read_excel(me_annotation_path)

output_annotations_dirpath = Path('../../../CASME3/processed_annotations')
output_annotations_dirpath.mkdir(parents=True, exist_ok=True)

output_annotations_filepath = output_annotations_dirpath / 'me_annotations.csv'

l = []

for index, row in tqdm(me_annotation.iterrows(), total=me_annotation.shape[0]):
    subject = row['Subject']
    filename = row['Filename']
    onset = row['Onset']
    apex = row['Apex']
    offset = row['Offset']
    action_unit = row['AU']
    objective = row['Objective class']
    emotion = row['emotion']

    datapoint_dirpath = data_dirpath / subject / filename
    # print(f'onset : {onset}, apex : {apex}, offset : {offset}')
    rgb_dirpath = datapoint_dirpath / 'color'
    depth_dirpath = datapoint_dirpath / 'depth'
    frames, onset, apex, offset = get_frames(rgb_dirpath, depth_dirpath, onset, apex, offset, 16)
    # print(frames)
    rgb_frames = np.array([np.array(Image.open(rgb_dirpath / f'{frame}.jpg')) for frame in frames])
    depth_frames = np.array([np.array(Image.open(depth_dirpath / f'{frame}.png')) for frame in frames])





    #Face Alignment and Cropping of images and depth maps
    rgb_frames_final =[]
    depth_frames_final = []
    h, w = rgb_frames[0].shape[:2]

    for i in range(len(rgb_frames)):
        frame_aligned, depth_frame_aligned = face_aligner(rgb_frames[i], depth_frames[i] )
        temp1, temp2 = face_cropper(frame_aligned, depth_frame_aligned)

        temp2 = temp2.squeeze()
        rgb_frames_final.append(temp1)
        depth_frames_final.append(temp2)
        # img = Image.fromarray(temp2)
        # img.show()
        # print(1/0)



    rgb_frames_final, depth_frames_final = pad_frames(rgb_frames_final, depth_frames_final)


    #Save the frames
    output_dirname = f'{subject}_{filename}_{onset}_{offset}'
    output_dirpath = ME_output_dirpath / output_dirname

    output_rgb_dirpath = output_dirpath / 'rgb'
    output_depth_dirpath = output_dirpath / 'depth'

    output_rgb_dirpath.mkdir(parents=True, exist_ok=True)
    output_depth_dirpath.mkdir(parents=True, exist_ok=True)
    for i in range(len(rgb_frames_final)):
        Image.fromarray(rgb_frames_final[i]).save(output_rgb_dirpath / f'{i}_{frames[i]}.jpg')
        np.save(output_depth_dirpath / f'{i}_{frames[i]}.npy', depth_frames_final[i])

    l.append([subject, filename, onset, apex, offset, action_unit, objective, emotion])

#Make a dataframe and save it
df = pd.DataFrame(l, columns=['subject', 'filename', 'onset', 'apex', 'offset', 'AU', 'objective_class', 'emotion'])
df.to_csv(output_annotations_filepath, index=False)
        