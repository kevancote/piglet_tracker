#script to run through video folder and perform pose analysis on goats

#be sure to run this in anaconda env "conda activate DLC-GPU"

import deeplabcut
import os

base_path = '.' 
config_path = os.path.join(base_path,'config.yaml')
#out_folder = os.path.join(base_path,'Out_Videos')
out_folder = '/Out_Videos'
in_folder = os.path.join(base_path,'In_Videos')


videos = os.listdir(in_folder)

for video in videos:
	deeplabcut.analyze_videos(config_path, [os.path.join(in_folder,video)], videotype='mp4',gputouse=1,save_as_csv=True,destfolder=out_folder)

