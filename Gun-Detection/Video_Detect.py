import time
import datetime
import cv2
import argparse
import numpy as np

# Cai dat tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# Ham ve khung chu nhat quanh vat the nhan dang
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detect_gun(net,output_layers,image):

    # Lay kich thuoc anh dau vao
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # Doc du lieu anh va dua vao mang YOLO
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.2
    nms_threshold = 0.1
    # confidences_threshhold

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        #print(x, y, w, h)

    return image

# read parameter
classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# setup net
net = cv2.dnn.readNet(args.weights, args.config)
output_layers = get_output_layers(net)


# Read from video file
url_path = args.image
cap = cv2.VideoCapture(url_path)
# out = cv2.VideoWriter(path,fourcc, 20, (460,360))

# define a box of Roid
frame_number = 0
car_counting = 0
objectID = 0
frame_number = 0
frame = None
prev_frame = None
while (cap.isOpened()):
    #start_time = time.time()
    ret_val, frame = cap.read()
    if frame is None:
        break
    # if prev_frame is not None:
    #     # --- take the absolute difference of the images ---
    #     res = cv2.absdiff(frame, prev_frame)
    #     # --- convert the result to integer type ---
    #     res = res.astype(np.uint8)
    #     # --- find percentage difference based on number of pixels that are not zero ---
    #     percentage = (np.count_nonzero(res) * 100) / res.size

    #     if (percentage>10):
    #         frame = detect_gun(net, output_layers, frame)
    # else:
    frame = detect_gun(net, output_layers, frame)

    # prev_frame = frame
    # frame_number = frame_number + 1

    cv2.putText(frame, str(frame_number), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()