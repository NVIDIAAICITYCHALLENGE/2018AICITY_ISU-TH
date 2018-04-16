import os

videos = os.listdir('track1_videos/')

for video in videos:
	frame_folder = 'track1_frames/' + video.split('.mp4')[0]
	if not os.path.exists(frame_folder):
		os.mkdir(frame_folder)

	video_source = 'track1_videos/' + video
	frame_dest = frame_folder + '/image%d.jpg'

	command = 'ffmpeg -i '+video_source+' -q:v 1 '+frame_dest+' -hide_banner'

	os.system(command)