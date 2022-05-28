# EEE 511 Project: Voxel based object detection and 3D reconstruction

# Team Members:
# 1) Rikenkumar Patel
# 2) Darsh Shah
# 3) Devang Kakadiya
# 4) Md Ragib Shaharea


# You need to first install Mediapipe using
# pip install mediapipe


# Import Library
import cv2
import time
import numpy as np
#import mediapipe as mp

#mp_objectron = mp.solutions.objectron
#mp_drawing = mp.solutions.drawing_utils

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Make list of classes
classes_yolo = []
with open("coco.name", "r") as f:
    classes_yolo = [line.strip() for line in f.readlines()]

# Load Network
layer = net.getLayerNames()
# print(layer)       #Print layers name

outputlayers = [layer[i - 1] for i in net.getUnconnectedOutLayers()]


# Define a function to detect object from image


def Detection(image):
    class_ids = []
    Get_confidences = []
    Get_Boxes = []
    Height, Width, Channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)
    for out in outs:
        for detect in out:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # object detected
                x_center = int(detect[0] * Width)
                y_center = int(detect[1] * Height)
                W = int(detect[2] * Width)
                H = int(detect[3] * Height)

                # rectangle co-ordinaters
                X = int(x_center - W / 2)
                Y = int(y_center - H / 2)
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                Get_Boxes.append([X, Y, W, H])  # put all rectangle areas
                Get_confidences.append(
                    float(confidence))  # how confidence was that object detected and show that percentage
                class_ids.append(class_id)  # name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(Get_Boxes, Get_confidences, 0.4, 0.6)

    for i in range(len(Get_Boxes)):
        if i in indexes:
            X, Y, W, H = Get_Boxes[i]
            label = str(classes_yolo[class_ids[i]])
            confidence = Get_confidences[i]
            cv2.rectangle(frame, (X, Y), (X + W, Y + H), (0, 0, 255), 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (X, Y + 30), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 2)

    Elapsed_Time = time.time() - Starting_time
    FPS = frame_id / Elapsed_Time

    # cv2.putText(frame, "FPS:" + str(round(FPS, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)
    cv2.imshow("Image", frame)


# Live Video Capture

Cap = cv2.VideoCapture(0)
Starting_time = time.time()

while True:
    # frame = cv2.imread("dog.jpg")  # Detect objects from image
    _, frame = Cap.read()
    frame_id = 0
    frame_id += 1
    Detection(frame)
    key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame

    if key == 27:  # esc key stops the process
        break;

#Cap.release()
cv2.destroyAllWindows()

# 3D Bounding Box

# Load the Image

# import glob

# Chair_images = {name: cv2.imread(name) for name in glob.glob("/content/R.jpg")}

# # Preview the images.
# for name, image in Chair_images.items():
#     cv2.imshow(image)

# with mp_objectron.Objectron(
#         static_image_mode=True,
#         max_num_objects=5,
#         min_detection_confidence=0.5,
#         model_name='Chair') as objectron:
#     # Run inference on shoe images.
#     for name, image in Chair_images.items():
#         # Convert the BGR image to RGB and process it with MediaPipe Objectron.
#         image = cv2.resize(image, (700, 700))
#         results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#         # Draw box landmarks.
#         if not results.detected_objects:
#             print(f'No box landmarks detected on {name}')
#             continue
#         print(f'Box landmarks of {name}:')
#         annotated_image = image.copy()
#         for detected_object in results.detected_objects:
#             mp_drawing.draw_landmarks(
#                 annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
#             mp_drawing.draw_axis(annotated_image, detected_object.rotation, detected_object.translation)
#         cv2.imshow(annotated_image)
