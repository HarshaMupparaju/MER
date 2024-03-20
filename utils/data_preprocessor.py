from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import glob

database_dirpath = Path('/media/harshamupparaju/Expansion/Harsha/Databases/CASME3/part_A/data/Compressed_version1_seperate_compress')

#List the folders in database
subjects = [i for i in database_dirpath.iterdir() if i.is_dir()]

for subject in subjects:
    subject_dirpath = database_dirpath / subject

    #List the folders in subject
    videos = [i for i in subject_dirpath.iterdir() if i.is_dir()]
    for video in videos:
        video_dirpath = subject_dirpath / video
        video_color_dirpath = video_dirpath / 'color'
        video_depth_dirpath = video_dirpath / 'depth'
        video_color_frames = [i for i in video_color_dirpath.iterdir() if i.is_file()]
        video_depth_frames = [i for i in video_depth_dirpath.iterdir() if i.is_file()]
        