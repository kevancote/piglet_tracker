#run through pose data gathered from videos and determine piglet interactions

import os
import pandas as pd
import cv2
import easygui
import math
import time
import numpy as np
import csv

base_path = '.'

#set global list for calibration points to be used in mouse callback
calib_points = []

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)
		calib_points.append(x*2)
		calib_points.append(y*2)

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, str(x) + ',' +
					str(y), (x,y), font,
					1, (255, 0, 0), 2)

		cv2.imshow('image', frame)

def isInside(circle_x, circle_y, rad, x, y):
     
    # Compare radius of circle
    # with distance of its center
    # from given point
    if ((x - circle_x) * (x - circle_x) +
        (y - circle_y) * (y - circle_y) <= rad * rad):
        return True;
    else:
        return False;

# driver function
if __name__=="__main__":

	#create output dictionary to write all results to
	output_df=pd.DataFrame()
	output_df_break=pd.DataFrame()
	pose_files_dir = easygui.diropenbox('Select folder containing videos')

	#include catch for selecting directory containing all directories or just the individual directory
	for foldername in os.listdir(pose_files_dir):
		#check if path contains more directories
		if os.path.isdir(os.path.join(pose_files_dir, foldername)):
			list_dirs = os.listdir(pose_files_dir)
		#if not then create a list with only one directory name
		else: list_dirs = [pose_files_dir]

	for dirs in list_dirs:
		output_dict = {'Trial_number':[], 'Piglet_id':[], 'Test':[], 'Distance_moved':[], 'Velocity_centre_mean':[], 'Velocity_centre_max':[],'Object_1_visits':[], 'Object_2_visits':[], 'Object_1_time':[], 'Object_2_time':[], 'Object_1_latency':[],'Object_2_latency':[]}
		pose_files = []
		pose_videos = []
		calib_points = []
		for files in os.listdir(os.path.join(pose_files_dir, dirs)):
			if files.endswith('.mp4'):
				pose_videos.append(os.path.join(pose_files_dir, dirs, files))
				for pose in os.listdir((os.path.join(base_path, 'Out_Videos'))):
					if pose.startswith(files[:-4]) and pose.endswith('.csv'):
						pose_files.append(os.path.join(base_path, 'Out_Videos', pose))
		pose_files.sort()

		if os.path.isfile(os.path.join(pose_files_dir, dirs, 'config.csv')):
			#read in calibration file in format: [calibration points(8 points (x,y) - pen top, pen bottom, object 1 centre, object 2 centre), pen_measurement(1 point), object_buffer_rad_cm(1 point)]
			with open(os.path.join(pose_files_dir, dirs, 'config.csv')) as f:
				reader = csv.reader(f)
				config = list(reader)
			calib_points = [eval(x) for x in config[0][:8]]
			print(calib_points)
			pen_distance = int(config[0][8])
			object_buffer_rad_cm = int(config[0][9])

		else:
			
			#capture mouse click points for calibration of pen size
			#open the first video and capture start image
			open_image = pose_videos[0]
			print(open_image)
			cap = cv2.VideoCapture(open_image)
			if (cap.isOpened()== False):
				print("Error opening video stream or file")
			cap.set(1,1)
			
			ret,frame = cap.read()
			(h,w) = frame.shape[:2]
			frame = cv2.resize(frame, (800,600))
			cv2.namedWindow('Pen', cv2.WINDOW_NORMAL)

			#cv2.setWindowProperty('Pen', cv2.WINDOW_FREERATIO, cv2.WINDOW_FULLSCREEN)
			# select points as: pen top, pen bottom, object 1, object 2
			cv2.imshow('image', frame)

			# setting mouse handler for the image
			# and calling the click_event() function
			cv2.setMouseCallback('image', click_event)
			# wait for a key to be pressed to exit
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			cap.release()
			print(calib_points)
			
			#hard code calib points and pen distance for testing
			#calib_points = [786, 128, 792, 1072, 564, 828, 568, 356]
			#calibration for pen
			pen_distance = easygui.enterbox(msg='Enter the width of the pen in cm')
			#pen_distance = 200
			#set buffer zone for objects
			object_buffer_rad_cm = easygui.enterbox(msg='Enter the buffer zone distance from object in cm')

		a_dist = abs(calib_points[0] - calib_points[2])
		b_dist = abs(calib_points[1] - calib_points[3])
			
		#units of cm/px, multiply with px to get the value in cm
		pen_px_to_cm = int(pen_distance) / math.sqrt((a_dist * a_dist) + (b_dist * b_dist))
		print(pen_px_to_cm)
		
		#object_buffer_rad_cm = 20
		object_buffer_rad_px = int(object_buffer_rad_cm) / pen_px_to_cm

		for trial_num, files in enumerate(pose_files):
			print(files)
			raw_df = pd.read_csv(files, skiprows=2)
			#rename headers
			raw_df.columns.values[0:] = ['Frame', 'Head_X', 'Head_Y', 'Head_likelihood', 'Body_X', 'Body_Y', 'Body_likelihood', 'Tail_X', 'Tail_Y', 'Tail_likelihood',]
			#print(raw_df.head())

			start_buffer = 30 * 5 #30 fps at 5 second buffer
			#loop through first frame and remove before piglet is detected plus X seconds
			for frame in raw_df['Frame'].tolist():
				if raw_df.iloc[frame]['Head_likelihood'] > 0.8:
					first_detect = frame
					break
			start_frame = first_detect + start_buffer
			################## Set start_frame to 0 to use the entire length of video for comparison ####################
			#start_frame = 0
			#print(start_frame)

			#Look only at head data for now, filter out low likelihood points with NaN
			head_df = raw_df[['Head_X','Head_Y','Head_likelihood', 'Body_X', 'Body_Y']]
			head_df = head_df.mask(head_df['Head_likelihood'] < 0.8)
			filter_df = head_df.join(raw_df['Frame'])

			#filter out first X amount of data depending on handler/ animal in pen
			#filter down data for only every X frame
			out_df = pd.DataFrame()
			
			#add columns for touching objects to filter_df
			touching_1 = [0] * len(filter_df)
			touching_2 = [0] * len(filter_df)
			filter_df['Touching 1'] = touching_1
			filter_df['Touching 2'] = touching_2

			#look for head point within object zone, only play for 5 mins of video
			end_frame = start_frame + (30*60*5)
			for frame in range(start_frame, end_frame):
				if math.isnan(filter_df.iloc[frame]['Head_X']):
					continue
				x = int(filter_df.iloc[frame]['Head_X'])
				y = int(filter_df.iloc[frame]['Head_Y'])

				if(isInside(int(calib_points[4]), int(calib_points[5]), int(object_buffer_rad_px), x, y)):
					#touching object 1 (first selected object)
					filter_df.at[frame, 'Touching 1'] = 1
				if(isInside(calib_points[6], calib_points[7], object_buffer_rad_px, x, y)):
					#touching object 2 (first selected object)
					filter_df.at[frame, 'Touching 2'] = 1
			
			#filter_df.to_csv('filter.csv')

			#filter data to 5 fps
			#use majority of touching 1, 2 or none to average data
			filter_fps = 5
			#convert fps value to frames per 30 frames original
			filter_fps = int(30 / filter_fps)
			fps_dict = {'Frame':[], 'original_frame':[], 'touching_1':[], 'touching_2':[], 'body_x':[], 'body_y':[]}
			for new_frame, frame in enumerate(range(start_frame, end_frame-filter_fps, filter_fps)):
				touch_1 = 0
				touch_2 = 0
				for i in range(frame, frame+filter_fps):
					if filter_df.iloc[i]['Touching 1']:
						touch_1 += 1
					if filter_df.iloc[i]['Touching 2']:
						touch_2 +=1
				if touch_1 > (filter_fps/2):
					touching_1_tot = 1
				else: touching_1_tot = 0
				if touch_2 > (filter_fps/2):
					touching_2_tot = 1
				else: touching_2_tot = 0
				#loop through the filtered frames and average the body point for the frames together for x and y
				body_x_list = []
				body_y_list = []
				for i in range(0, filter_fps):
					body_x_list.append(filter_df.iloc[frame+i]['Body_X'])
					body_y_list.append(filter_df.iloc[frame+i]['Body_Y'])
				body_x_avg = sum(body_y_list)/len(body_y_list)
				body_y_avg = sum(body_y_list)/len(body_y_list)

				#add in filter for a 1 second change to be detected before recording a change in state of objects

				#add all the calculated points to a dictionary which will convert to dataframe
				fps_dict['body_x'].append(body_x_avg)
				fps_dict['body_y'].append(body_y_avg)
				fps_dict['Frame'].append(new_frame)
				fps_dict['touching_1'].append(touching_1_tot)
				fps_dict['touching_2'].append(touching_2_tot)
				fps_dict['original_frame'].append(frame)

			sum_df = pd.DataFrame.from_dict(fps_dict)
			sum_df.to_csv('filter_sum.csv')

			#create new data frame to match the gold standard output file (Object interactions with times)
			sum_dict = {'Piglet_id':[], 'Test':[], 'Frame':[], 'original_frame':[], 'object':[], 'time_on':[], 'time_off':[]}
			
			#loop through for object 1 
			for frame in range(1, len(sum_df)-1):
				#check if piglet is touching initially on object
				if int(sum_df.iloc[frame]['touching_1']) == 1 and frame == 1:
					sum_dict['Frame'].append(frame)
					sum_dict['original_frame'].append(sum_df.iloc[frame]['original_frame']) 
					sum_dict['object'].append(1)
					sum_dict['time_on'].append(float(sum_df.iloc[frame]['original_frame']/30))
					continue
				#start touching 1 when frame is recorded as touching
				elif int(sum_df.iloc[frame]['touching_1']) == 1 and int(sum_df.iloc[frame-1]['touching_1']) == 0:
					sum_dict['Frame'].append(frame)
					sum_dict['original_frame'].append(sum_df.iloc[frame]['original_frame']) 
					sum_dict['object'].append(1)
					sum_dict['time_on'].append(float(sum_df.iloc[frame]['original_frame']/30))
					continue
				elif int(sum_df.iloc[frame]['touching_1']) == 0 and int(sum_df.iloc[frame-1]['touching_1']) == 1 and not frame == 1:
					sum_dict['time_off'].append(float(sum_df.iloc[frame]['original_frame']/30))
				#skip any frames where both are not touching
				#order of last two elif statements is important, could get triggered early if before previous statements and miss off times
				elif int(sum_df.iloc[frame]['touching_1']) == 0:
					continue
				#skip frames where current and next frame are touching	
				elif int(sum_df.iloc[frame]['touching_1']) == 1 and int(sum_df.iloc[frame+1]['touching_1']) == 1: 
					continue
			if int(sum_df.iloc[len(sum_df)-1]['touching_1']) == 1 and int(sum_df.iloc[len(sum_df)-2]['touching_1']) == 1:
				sum_dict['time_off'].append(float(sum_df.iloc[len(sum_df)-1]['original_frame']/30))
			elif int(sum_df.iloc[len(sum_df)-1]['touching_1']) == 0 and int(sum_df.iloc[len(sum_df)-2]['touching_1']) == 1:
				sum_dict['time_off'].append(float(sum_df.iloc[len(sum_df)-1]['original_frame']/30))
			#if int(sum_df.iloc[0]['touching_1']) == 1:
			#	sum_dict['time_on'].append(float(sum_df.iloc[0]['original_frame']/30))
				
			#loop through again for object 2
			for frame in range(1, len(sum_df)-1):
				#check if initial frame is touching object
				if int(sum_df.iloc[frame]['touching_2']) == 1 and frame == 1:
					sum_dict['Frame'].append(frame)
					sum_dict['original_frame'].append(sum_df.iloc[frame]['original_frame']) 
					sum_dict['object'].append(2)
					sum_dict['time_on'].append(float(sum_df.iloc[frame]['original_frame']/30))
					continue
				#start touching 2 when frame is recorded as touching
				elif int(sum_df.iloc[frame]['touching_2']) == 1 and int(sum_df.iloc[frame-1]['touching_2']) == 0:
					sum_dict['Frame'].append(frame)
					sum_dict['original_frame'].append(sum_df.iloc[frame]['original_frame']) 
					sum_dict['object'].append(2)
					sum_dict['time_on'].append(float(sum_df.iloc[frame]['original_frame']/30))
					continue
				elif int(sum_df.iloc[frame]['touching_2']) == 0 and int(sum_df.iloc[frame-1]['touching_2']) == 1 and not frame == 1:
					sum_dict['time_off'].append(float(sum_df.iloc[frame]['original_frame']/30))
				#skip any frames where both are not touching
				#order of last two elif statements is important, could get triggered early if before previous statements and miss off times
				elif int(sum_df.iloc[frame]['touching_2']) == 0:
					continue
				#skip frames where current and next frame are touching	
				elif int(sum_df.iloc[frame]['touching_2']) == 1 and int(sum_df.iloc[frame+1]['touching_2']) == 1: 
					continue
			if int(sum_df.iloc[len(sum_df)-1]['touching_2']) == 1 and int(sum_df.iloc[len(sum_df)-2]['touching_2']) == 1:
				sum_dict['time_off'].append(float(sum_df.iloc[len(sum_df)-1]['original_frame']/30))
			elif int(sum_df.iloc[len(sum_df)-1]['touching_2']) == 0 and int(sum_df.iloc[len(sum_df)-2]['touching_2']) == 1:
				sum_dict['time_off'].append(float(sum_df.iloc[len(sum_df)-1]['original_frame']/30))
			#if int(sum_df.iloc[0]['touching_2']) == 1:
			#	sum_dict['time_on'].append(float(sum_df.iloc[0]['original_frame']/30))
			
			#calculate distance moved and velocity
			distance = []
			velocity = []
			for i in range(0,len(sum_df)-1):
				x1 = sum_df.iloc[i]['body_x']
				x2 = sum_df.iloc[i+1]['body_x']
				y1 = sum_df.iloc[i]['body_y']
				y2 = sum_df.iloc[i+1]['body_y']
				distance_moved = math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
				distance.append(distance_moved)
				velocity.append(distance_moved/(1/filter_fps))
			sum_df = sum_df.join(pd.DataFrame(distance, columns = ['distance']))
			sum_df = sum_df.join(pd.DataFrame(velocity, columns = ['velocity']))
			total_distance = np.nansum(sum_df['distance'])*pen_px_to_cm
			max_velocity = np.nanmax(sum_df['velocity'])*pen_px_to_cm
			mean_velocity = np.nanmean(sum_df['velocity'])*pen_px_to_cm
			
			#split up file name to get trail number, id and test 
			output_dict['Trial_number'].append('Trial ' + str(trial_num+1))
			id_loc = files.split('/')[-1].find('Pig')
			pig_id = files.split('/')[-1][id_loc+3:id_loc+6]
			output_dict['Piglet_id'].append(pig_id)
			test = 'NOR'
			output_dict['Test'].append(test)

			#create columns which are repeated values for id and test the length of the dictionary
			sum_dict['Piglet_id'] = np.resize([pig_id], len(sum_dict['Frame']))
			sum_dict['Test'] = np.resize([test], len(sum_dict['Frame']))
			print(len(sum_dict['Piglet_id']))
			print(len(sum_dict['Test']))
			print(len(sum_dict['Frame']))
			print(len(sum_dict['original_frame']))
			print(len(sum_dict['object']))
			print(len(sum_dict['time_on']))
			print(len(sum_dict['time_off']))


			sum_df2 = pd.DataFrame.from_dict(sum_dict)
			sum_df2 = sum_df2.sort_values(by=['Frame'])
			sum_df2 = sum_df2.reset_index(drop=True)

			sum_df2.to_csv('filter_sum2.csv')

			##### add in section to include a 1 second buffer on bout changing to match golden standard coding, to be added as a flag
			bout_buffer = 1

			#add column in sum_df2 to label rows to drop for bout buffering
			sum_df2['drop'] = [0] * len(sum_df2)
			
			#check for dataframes which could be too short
			if len(sum_df2) > 1:
				#check for detections shorter than bout buffer
				for i in range(0, len(sum_df2)):
					if sum_df2.iloc[i]['time_off'] - sum_df2.iloc[i]['time_on'] < bout_buffer:
						sum_df2.at[i, 'drop'] = 1
				#drop the rows based on being shorted than the bout buffer
				sum_df2 = sum_df2.drop(sum_df2.index[sum_df2['drop'] == 1].tolist())
				sum_df2 = sum_df2.reset_index(drop=True)
				
				#check for gaps longer than bout buffer between entries
				#start at second observation and look back
				j = 0
				for i in range(1, len(sum_df2)):
					if sum_df2.iloc[i]['object'] == sum_df2.iloc[i-1]['object'] and (sum_df2.iloc[i]['time_on'] - sum_df2.iloc[i-1]['time_off']) < bout_buffer:
						sum_df2.at[i, 'drop'] = 1
						if sum_df2.iloc[i]['object'] == sum_df2.iloc[i-1]['object'] and sum_df2.iloc[i-1]['drop'] == 0:
							sum_df2.at[i-1, 'time_off'] = sum_df2.iloc[i]['time_off']
						elif sum_df2.iloc[i]['object'] == sum_df2.iloc[i-1]['object'] and sum_df2.iloc[i-1]['drop'] == 1:
							j += 1
							sum_df2.at[i-1-j, 'time_off'] = sum_df2.iloc[i]['time_off']	
					if sum_df2.iloc[i-1]['drop'] == 1 and (sum_df2.iloc[i]['time_on'] - sum_df2.iloc[i-1]['time_off']) > bout_buffer:
						j = 0
				sum_df2 = sum_df2.drop(sum_df2.index[sum_df2['drop'] == 1].tolist())

			sum_df2 = sum_df2.reset_index(drop=True)

			print(sum_df2)

			touch_1_list = sum_df2.index[sum_df2['object'] == 1].tolist()
			touch_1_count = len(touch_1_list)

			touch_2_list = sum_df2.index[sum_df2['object'] == 2].tolist()
			touch_2_count = len(touch_2_list)

			object_1_time = 0
			for i in touch_1_list:
				object_1_time += (sum_df2.iloc[i]['time_off'] - sum_df2.iloc[i]['time_on'])

			object_2_time = 0
			for i in touch_2_list:
				object_2_time += (sum_df2.iloc[i]['time_off'] - sum_df2.iloc[i]['time_on'])

			if len(touch_1_list) > 0:
				latency_1 = sum_df2.iloc[touch_1_list[0]]['time_on']
			else: latency_1 = 'NA'
			if len(touch_2_list) > 0:
				latency_2 = sum_df2.iloc[touch_2_list[0]]['time_on']
			else: latency_2 = 'NA'
				
			#create output dataframe
			output_dict['Distance_moved'].append(total_distance)
			output_dict['Velocity_centre_mean'].append(mean_velocity)
			output_dict['Velocity_centre_max'].append(max_velocity)
			output_dict['Object_1_visits'].append(touch_1_count)
			output_dict['Object_2_visits'].append(touch_2_count)
			output_dict['Object_1_time'].append(object_1_time)
			output_dict['Object_2_time'].append(object_2_time)
			output_dict['Object_1_latency'].append(latency_1)
			output_dict['Object_2_latency'].append(latency_2)


			if not output_df_break.empty:
				output_df_break = pd.concat([output_df_break, sum_df2], ignore_index=True)
			else:
				output_df_break = sum_df2

		if not output_df.empty:
			print('output df found')
			temp_df = pd.DataFrame.from_dict(output_dict)
			#temp_df = temp_df.sort_values(by=['Piglet_id'])
			output_df = pd.concat([output_df, temp_df], ignore_index=True)
			
		else: 
			print('creating output df')
			output_df = pd.DataFrame.from_dict(output_dict)
			output_df.sort_values(by=['Piglet_id'])

			

	output_df.to_csv('output.csv')
	output_df_break.to_csv('output_breakdown.csv')
