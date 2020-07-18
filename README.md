# Gun-Detection-In-Photos-Videos
- Detect guns in photos and in videos using Yolov3
## Preprocess:
### Download Pretrained Convolutional Weights:
For training I use convolutional weights that are pre-trained on Imagenet. I use weights from the darknet74 model. You can just download the weights for the convolutional layers [here](https://drive.google.com/file/d/1-oHpg4jBsjAQVlB2anXYFuOU7Dd3tNG7/view).
### Preparing YOLOv3 configuration files:
YOLOv3 needs certain specific files to know how and what to train. I’ll be creating these three files(.data, .names and .cfg) and also edit the yolov3.cfg.
- First let’s prepare the YOLOv3 .data and .names file. Let’s start by creating yolo.data and filling it with this content. This basically says that we are training one class, what the train and validation set files are and what file contains the names for the categories we want to detect.
```
classes= 1
train  = train.txt
valid  = val.txt
names = yolo.names
backup = backup
```
- *The backup is where you want to store the yolo weights file.* The yolo.names looks like this, plain and simple. Every new category should be on a new line, its line number should match the category number in the .txt label files i created earlier.
```
Gun
```
- Now i go to create the .cfg for choose the yolo architecture. I just duplicated the yolov3.cfg file, and made the following edits: 
Change the Filters and classes value.
```
+ Line 603: set filters=(classes + 5)*3 in my case filters=21
+ Line 610: set classes=2, the number of categories i want to detect
+ Line 689: set filters=(classes + 5)*3 in my case filters=21
+ Line 696: set classes=2, the number of categories i want to detect
+ Line 776: set filters=(classes + 5)*3 in my case filters=21
+ Line 783: set classes=2, the number of categories i want to detect
```
## Training:
Weights only save every 100 iterations until 900, then saves every 10,000. If you want change the process please follow the [link](https://github.com/pjreddie/darknet/issues/190).
```
./darknet detector train yolo.data cfg/yolov3.cfg darknet53.conv.74
```
## Demo:
<p align="center"> <img src="https://github.com/manhminno/Gun-Detection-In-Photos-Videos/blob/master/Gun-Detection/object-detection.jpg"></p>
<p align="center"> <img src="https://github.com/manhminno/Gun-Detection-In-Photos-Videos/blob/master/Gun-Detection-Yolov3-Using-Pytorch/Result.jpg"></p>

## Reference: 
- [Darknet](https://pjreddie.com/darknet/yolo/)
- [Mì AI](https://www.miai.vn/2019/08/09/yolo-series-2-cach-train-yolo-de-detect-cac-object-dac-thu/)
