# Gun Detection Using Yolov3 - Pytorch: Real-time detection
- U can download weight i had trained about 12 hours on GTX 1080 Ti: https://drive.google.com/open?id=10U0HetPuULgAxmnboQIAyFEyWMuU01Gc
- Run code:
```
*For webcam detect: python detect.py --weights weights/last.pt --name yolo.names --cfg cfg/yolov3.cfg --source 0
*For video, image: python detect.py --weights weights/last.pt --name yolo.names --cfg cfg/yolov3.cfg --source (Img/Video_file)
```
- Demo:
<p align="center"> <img src="https://github.com/manhminno/Gun-Detection-In-Photos-Videos/blob/master/Gun-Detection-Yolov3-Using-Pytorch/Result.jpg"></p>
