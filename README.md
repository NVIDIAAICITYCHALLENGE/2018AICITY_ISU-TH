# 2018AICITY_ISU-TH

## Dependency

Tensorflow Object Detection API depends on the following libraries:

* Protobuf 2.6
* Python-tk
* Pillow 1.0
* lxml
* tf Slim (which is included in the "tensorflow/models/research/" checkout)
* Jupyter notebook
* Matplotlib
* Tensorflow
* Cython
* cocoapi

For detailed steps to install Tensorflow, follow the Tensorflow installation instructions: https://www.tensorflow.org/install/

Additional dependency:

* cv2

## Instruction

`all_p` is used to store all the pickles of detection results, tracking results and conversion matrices. Existing results can be downloaded here: https://drive.google.com/open?id=1gaTrK-odF7ysPovTnqI2a1pVSAVvGy-o

1. Extract frames: `frame_extract.py`, create `track1_frames` folder first.
2. Detection: `detect.py`, results saved in `all_p/detect_p/`
3. Tracking: `track.py`, results saved in `all_p/track_p/`
4. Warping: `WARPING.ipnb`, individual warping for each video, results saved as `all_p/conversion1.p` and `all_p/warpM1.p`
5. Speed conversion: `HIGHWAY.py` for location 1 and 2, `INTERSECTION.py` for location 3 and 4. Results saved in `res/`
6. Merge files: `GenerateResults.py`, generate file `track1.txt`