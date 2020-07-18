## Original code from darknet:
- Detect guns in photos and in videos using Yolov3.
- I trained the model using google colab, trained 4,000 epochs.
- I used the darknet source code and conducted editing and training of my own model.
### *You can download weights that I've trained here: [Download](https://drive.google.com/open?id=1hdn6ndSQbAAEByIXoxB3tCu68st0h6Wc)*
### *You can download pre-train of darknet here: [Download](https://drive.google.com/file/d/1-oHpg4jBsjAQVlB2anXYFuOU7Dd3tNG7/view)*
### *Read more about Yolo: https://pjreddie.com/darknet/yolo/*
### Command run:
```
* For image detection: python YOLO.py -i (your_image) -cl (your_classes) -w (your_weight) -c (your_cfgfile)
* For video detection: python Video_Detect.py -i (your_video) -cl (your_classes) -w (your_weight) -c (your_cfgfile)
```
### Demo:
<p align="center"> 
<img src="https://github.com/manhminno/Gun-Detection-In-Photos-Videos/blob/master/Gun-Detection/object-detection.jpg">
</p>
