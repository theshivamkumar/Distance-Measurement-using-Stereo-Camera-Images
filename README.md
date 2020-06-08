# Objects Recognitiona and Distance Measurement using Stereo Camera Images
> Object detection and distance measurement in stereo images. Can be used as an aid for the visually impaired.

[![License](https://img.shields.io/github/license/theshivamkumar/Distance-of-objects-in-stereo-images.svg)](https://github.com/theshivamkumar/Distance-of-objects-in-stereo-images/blob/master/LICENSE)
[![Issues](https://img.shields.io/github/issues/theshivamkumar/Distance-of-objects-in-stereo-images.svg)](https://github.com/theshivamkumar/Distance-of-objects-in-stereo-images/issues)
[![Forks](https://img.shields.io/github/forks/theshivamkumar/Distance-of-objects-in-stereo-images.svg)](https://github.com/theshivamkumar/Distance-of-objects-in-stereo-images/network/members)
[![Stars](https://img.shields.io/github/stars/theshivamkumar/Distance-of-objects-in-stereo-images.svg)](https://github.com/theshivamkumar/Distance-of-objects-in-stereo-images/stargazers)

This is a project that takes a pair of frames from a stereo camera, finds the most prominent object in the image, computes the approximate distance the object is in real life from the stereo camera and speaks the result out. Hence, if modelled into a device, this project can be used as an aid for visually impaired people, as it can speak to them what object is infront of them and at what distance. <br>
This was my final year B.Tech Project at IIT Jammu.

<br>

## Requirements
- Python 3.6.10
- Tensorflow 1.15.0
- Keras 2.3.1
- OpenCV_Contrib 3.4.2.16
- ImageAI 2.1.5
- Pyttsx3 2.87

</br>

## Set-Up and running
### Get the requirements
Set-up for Anaconda users (non-anaconda users may use `pip3` or python-3.6 pip):

```
conda create --name stereo_env python=3.6
conda activate stereo_env
pip install tensorflow==1.15 keras==2.3.1
pip install opencv-contrib-python==3.4.2.16
pip install imageai==2.1.5
pip install pyttsx3==2.87
```
and 
`sudo apt install espeak` (for text to speech engine).

### Clone the repository 

```
git clone https://github.com/theshivamkumar/Distance-of-objects-in-stereo-images.git
cd Distance-of-objects-in-stereo-images
```

### Merge the model-files

The YOLOv3 and Resnet model-files are splitted as github allows only 100mb as max file size. So, merge the files in `Distance-of-objects-in-stereo-images/Models/` folder. 

```
cd Models
cat yolo.h5_1 yolo.h5_2 yolo.h5_3 > yolo.h5
rm yolo.h5_1 yolo.h5_2 yolo.h5_3
cat resnet50_coco_best_v2.0.1.h5_1 resnet50_coco_best_v2.0.1.h5_2 > resnet50_coco_best_v2.0.1.h5
rm resnet50_coco_best_v2.0.1.h5_1 resnet50_coco_best_v2.0.1.h5_2
cd .. 
```

Windows users may use the `type` command instead of `cat` to merge files.

### Run the project

```bash
cd src
conda activate stereo_env  #only for the Anaconda users who followed the requirements step in this README
python main.py  #the Anaconda users who followed the requirements step can use this, others may use python3 (use python 3.6 to execute)
```
It will print the result and speak out also!

<br>


## Brief Theory

Here mainly two things are being done. The first one is object detection. I have used [ImageAI](https://github.com/OlafenwaMoses/ImageAI) for object detection. It supports YOLOv3, TinyYOLOv3 and Resnet50 models. The most promintent object is used on the particular frame-pair, and is determined by largest area.</br>
<p align="center">
  <img src="https://github.com/theshivamkumar/Distance-of-objects-in-stereo-images/blob/master/Examples_of_Object_Recognition_and_Feature_Matching/ObjectRecognition.png" alt="Object Detection"/>
</p>


The second thing is feature matching. After detecting an object, the common features in both the objects are detected and matched, for that I am using [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform). This gives me the coordinates of the pixels of the corresponding matches of both the images.</br><br>
<p align="center">
  <img src="https://github.com/theshivamkumar/Distance-of-objects-in-stereo-images/blob/master/Examples_of_Object_Recognition_and_Feature_Matching/FeatureMatching.png" alt="Feature Matching"/>
</p>

Finally to calculate the distance, I use the difference in x-coordinates of the pixels of the matched features. Intuitively, the more the difference, the nearer the object and vice-versa. 

The report in the `Doc` folder has the complete details of this project, explaining exactly how to measure the distance. <br>
There is a variable called `fd`, it is different for different sets of stereo cameras. To find your `fd`, you would reqire the stereo image of an object, prominent in the frame and at a known distance from your stereo camera. You can calculate `fd` as, `fd = (Distance of object from the stereo camera) x (Difference in x-coordinates of the pixels of the matching points in the object in both the images)`. This whole project is under assumption that the stereo camera set would be exactly horizontal, i.e., vertical height of both the cameras would be same.

</br>

## References
- https://www.youtube.com/watch?v=sW4CVI51jDY
- https://github.com/OlafenwaMoses/ImageAI
- https://medium.com/@pwc.emtech.eu/object-detection-with-imageai-106d584984e9
- https://www.youtube.com/watch?v=75YtldrfxBU
- https://lmb.informatik.uni-freiburg.de/resources/datasets/StereoEgomotion/
