# !pip install mediapipe

# !wget https://pjreddie.com/media/files/yolov3.weights

import cv2
import argparse
import numpy as np
import mediapipe as mp

classes = None

with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def get_output_layers(net):

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    if label == 'person':
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
        cv2.putText(img, label, (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if x > 0 and y > 0:
          crop_img = img[y:y_plus_h, x:x_plus_w]

          imgRGB = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
          results = pose.process(imgRGB)
          
          if results.pose_landmarks:
            mpDraw.draw_landmarks(crop_img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

          cv2.imshow('Video', img)


video=cv2.VideoCapture(0)

writer = None
(Width, Height) = (None, None)

while True:
    check, image = video.read()
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    blob = cv2.dnn.blobFromImage(
        image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_prediction(image, class_ids[i], confidences[i], 
                        round(x), round(y), round(x+w), round(y+h))

    if writer is None:
      fourcc = cv2.VideoWriter_fourcc(*"DIVX")
      writer = cv2.VideoWriter("OutputVideo.mp4", fourcc, 20,
        (Width, Height), True)

    writer.write(image)
    
    video.release()
    writer.release()
    cv2.destroyAllWindows()
    break

