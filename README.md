Place videos to be analyzed in the "In_Videos" folder

Ensure you have docker installed. If not refer to the docker documentation for installation on your system.
Download and copy the docker image linked here into the base folder of this project.
Run docker pull cotekevan/piglet_tracker:version1
To load the image into your docker local directory

once it has finished building the container run:
docker run --rm --gpus=all -v ./Out_Videos:/Out_Videos -v ./In_Videos:/app/In_Videos cotekevan/piglet_tracker:version1

This will analyze the video and save files to the "Out_Videos" directory. This includes the csv file breaking down each frame of the video to a pose position as well as the h5 model file and the video file

Any future videos can be placed into the "In_Videos" folder and the above docker run command will analyze them as well.

install the requirements from the requirements.txt file

run the python command:
python process_data.py

This will prompt you to answer a few questions:
'Select folder containing videos' will be where the video files were saved to. In the default case this is "Out_Videos"

'Enter the width of the pen in cm' is the distance from the piglets left to right as they enter the pen. Note: in our case this was the frame orientation top to bottom distance since the entry was from the right

'Enter the buffer zone distance from object in cm' is the buffer radius around each object to count as an interraction. In the paper we used 22 cm

An output csv will be created called output.csv which includes all of the reported statistics from the referenced paper. There is also an included output_breakdown.csv which lists each interaction point and the time at which it happened
