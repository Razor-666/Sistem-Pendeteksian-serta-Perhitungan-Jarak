from imutils import paths
import numpy as np
import imutils
import cv2
import os
import time
import math
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


#Path to saved model  

PATH_TO_SAVED_MODEL = "/home/fire-x/Documents/Data_Skripsi_Programming/distance/Hasil_671000steps_29Juni2023/inference_graph/saved_model"

# Load label map and obtain class names and ids
#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
category_index=label_map_util.create_category_index_from_labelmap("/home/fire-x/Documents/Data_Skripsi_Programming/distance/Hasil_671000steps_29Juni2023/label_map.pbtxt",use_display_name=True)

# Distance constants 
# KNOWN_DISTANCE = 20 #20INCHES atau cm==50/51
BALL_WIDTH = 20 #8INCHES JIKA CM==20

#selalu kalibrasi titik tengah frame yaitu titik tengah kamera ;v
def garis_pembagi(image):
    (h, w) = image.shape[:2] #w:image-width and h:image-height
    # print(h,w) #h=480;w=640
    #kamera ELP R2
    cv2.line(image, (0,h//2 + 10 ), (w,h//2+10), (0,0,255),2,8,0) #garis atas bawah kolom
    cv2.line(image, (w//2+20 ,0), (w//2 +20,h), (0, 0, 255), 2,8,0) #garis kanan kiri baris
    cv2.putText(image, "1", (620,25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)
    cv2.putText(image, "4", (315,25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)
    cv2.putText(image, "3", (315,280), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)
    cv2.putText(image, "2", (620,280), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)


def pembagi_piksel(image): #kamera depan jika mau omnidirectional dikalibrasi kembali
    cv2.putText(image, "1", (620,25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)
    cv2.putText(image, "4", (620,185), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)
    cv2.putText(image, "7", (620,345), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)

    cv2.putText(image, "2", (405,25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)
    cv2.putText(image, "5", (405,185), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)
    cv2.putText(image, "8", (405,345), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)

    cv2.putText(image, "3", (188,25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)
    cv2.putText(image, "6", (188,185), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)
    cv2.putText(image, "9", (188,345), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,255), 1)

    cv2.line(image, (213,0), (213,480), (0, 0, 255), 2,8,0) #garis atas kebawah
    cv2.line(image, (427,0), (427,480), (0, 0, 255), 2,8,0) #garis atas kebawah
    cv2.line(image, (0,160), (640,160), (0,0,255),2,8,0) #garis bawah keatas
    cv2.line(image, (0,320), (640,320), (0,0,255),2,8,0) #garis bawah keatas

#coba kamera depan deteksi bola
def ball_detect(image, bboxes, labels, scores, thresh):
    (h, w, d) = image.shape
    pembagi_piksel(image)
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)
            x, y = int((xmin + xmax ) / 2), int((ymin+ ymax) / 2) #menentukan titik tengah objek/benda deteksi

            (h, w) = image.shape[:2] # h = untuk vertical ; w= untuk horizontal
            center_x = int(w/2) #titik tengah frame 320
            center_y = int(h/2) #titik tengah frame 240

            # #mencari arah datang sudutnya
            # angle = int(math.atan2(h - y,center_x - x) * 180 / math.pi)
            # angle = angle - 90 if angle > 90 else angle + 270
            
            if labels == ['bola']:
                cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2 #center dalam citra bounding box
                length = math.hypot(xmax - xmin, ymax - ymin)
                # print(length)
                info = (xmin, ymin, xmax, ymax, cx, cy) #cx dan cy merupakan titik tengah benda
              
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(image, f"{label}: {int(score*100)} %", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                cv2.line(image,(center_x,480), (cx,cy), (0,255,255), 2, 8, 0)

                # cv2.line(image, (320,480), (cx,cy), (255, 255, 255), 2, 8, 0) #(w//2-30, h//2+36) //baris tengah jaraknya
                jarakbenda = int(math.sqrt(int((center_x-cx)**2) + int((center_y-cy)**2))) #dalam pixel
                cv2.putText(image, f'Jarak:{jarakbenda}px', (430,450), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 3)

                w=jarakbenda

                # focal_length= float((w* KNOWN_DISTANCE)/BALL_WIDTH)
                # print(focal_length)
                focal_length = 120 #165 == 30cm
                distance = (float((BALL_WIDTH * focal_length) / w))
                # print(distance)

                # distance1 = (2 * 3.14 * 180) / (center_x + center_y  * 360) * 1000 + 3
                # print(info,";",length,";",distance1)
                # if distance1 < 11:
                #      distance1 =0
                
                # print(focal_length, ";", ymin, ";", round(distance1), ";", round(distance))

                cv2.putText(image, f'Jarak:{round(distance)}cm', (430,420), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 3)

                # cv2.putText(image, f'{angle}', (320,470), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,0), 2)   

#omnidirectional
def visualise_on_image(image, bboxes, labels, scores, thresh):
    (h, w, d) = image.shape
    garis_pembagi(image)
    cv2.circle(image, ((w//2)+20,(h//2)+10), 5, (255, 0, 255), cv2.FILLED)
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)
            x, y = int((xmin + xmax ) / 2), int((ymin+ ymax) / 2) #menentukan titik tengah objek/benda deteksi

            (h, w) = image.shape[:2] # h = untuk vertical ; w= untuk horizontal
            center_x = int(w/2) #titik tengah frame 320
            center_y = int(h/2) #titik tengah frame 240

            #mencari arah datang sudutnya omnidirectional
            angle = int(math.atan2(h - y,center_x+20 - x) * 180 / math.pi)
            angle = angle - 90 if angle > 90 else angle + 270
            
            if labels == ['bola']:
                cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
                length = math.hypot(xmax - xmin, ymax - ymin)
                # print(length)
                info = (xmin, ymin, xmax, ymax, cx, cy) #cx dan cy merupakan titik tengah benda
              
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(image, f"{label}: {int(score*100)} %", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                cv2.line(image,(center_x+20,center_y+10), (cx,cy), (0,255,255), 2, 8, 0)

                # cv2.line(image, (320,480), (cx,cy), (255, 255, 255), 2, 8, 0) #(w//2-30, h//2+36) //baris tengah jaraknya
                #masih di cari buat pantulannya
                # jarakbenda = int(math.sqrt(int(((center_x-30)-cx)**2) + int(((center_y+45)-cy)**2))) #dalam pixel
                # print(jarakbenda)
                # cv2.putText(image, f'Jarak:{int(jarakbenda)}px', (430,450), cv2.FONT_HERSHEY_DUPLEX, 1, (50,0,255), 1)
                
                # w=jarakbenda

                # # focal_length= float((w* 40)/BALL_WIDTH)
                # # print(focal_length)
                # focal_length = 160 #165 == 30cm
                # distance = (float((BALL_WIDTH * focal_length) / w))
                # print(distance)



                # focal_length= float((cx * KNOWN_DISTANCE)/BALL_WIDTH)
                # distance = (float((BALL_WIDTH * focal_length) / xmax))
                # distance1 = (2 * 3.14 * 180) / (center_x + center_y  * 360) * 1000 + 3
                # print(info,";",length,";",distance1)
                # if distance1 < 11:
                #      distance1 =0
                
                # print(focal_length, ";", ymin, ";", round(distance1), ";", round(distance))
                # cv2.putText(image, f'Jarak:{round(distance)}cm', (430,390), cv2.FONT_HERSHEY_DUPLEX, 1, (50,0,255), 1)
                cv2.putText(image, f'{angle}', (320,470), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,0), 2)
                # print(int(distance*2.54))

            # cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
            # length = math.hypot(xmax - xmin, ymax - ymin)
            # # print(length)
            # info = (xmin, ymin, xmax, ymax, cx, cy) #cx dan cy merupakan titik tengah benda
            # print(info)
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            # cv2.putText(image, f"{label}: {int(score*100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # cv2.circle(image, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
            # cv2.line(image, (320,480), (cx,cy), (255, 255, 255), 2, 8, 0) #(w//2-30, h//2+36)
            # jarakbenda = int(math.sqrt(int((center_x-cx)**2) + int((center_y-cy)**2))) #dalam pixel
            # cv2.putText(image, f'Jarak: {int(jarakbenda)} cm', (400,350), cv2.FONT_HERSHEY_DUPLEX, 1, (50,0,255), 2)

            # cv2.line(image, (320,480), (x,y), (255, 255, 255), 2, 8, 0) #(w//2-30, h//2+36)
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            # cv2.putText(image, f"{label}: {int(score*100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # # cv2.circle(image,(x,y),5,(0,0,255),-1)
            # jarakbenda = int(math.sqrt(int((cx-x)**2) + int((cy-y)**2)))
            # cv2.putText(image, f'Jarak: {int(jarakbenda)}', (400,350), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 1)
            
    return image

if __name__ == '__main__':
    
    # Load the model
    print("[INFO] Loading saved model ...")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("[INFO] Model Loaded!")
    
    # video_capture = cv2.VideoCapture("/home/fire-x/Documents/Data_Skripsi_Programming/distance/coba_5.mp4")
    # video_capture = cv2.VideoCapture("/home/fire-x/Documents/Data_Skripsi_Programming/distance/Hasil-depan.mp4")
    video_capture = cv2.VideoCapture(2)
    video_capture_omni = cv2.VideoCapture(4)
    video_capture.set(3,640)
    video_capture.set(4,480)
    video_capture_omni.set(3,640)
    video_capture_omni.set(4,480)
    ##### REAL TIME ######
    # video_captur = cv2.VideoCapture(2)
    # video_capture.set(3,640)
    # video_capture.set(4,480)
    video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    video_capture.set(cv2.CAP_PROP_FOCUS, 52)
    video_capture_omni.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # video_capture_omni.set(cv2.CAP_PROP_FOCUS, 51)

    start_time = time.time()
    # set timer
    e1 = cv2.getTickCount()
    # proses setiap frame
    nframes = 0
    
    # frame_width = int(video_capture.get(3))
    # frame_height = int(video_capture.get(4))
    # fps = int(video_capture.get(5))
    # size = (frame_width, frame_height)
    # size = (640, 480)
    # # Initialize video writer
    # result = cv2.VideoWriter('/home/fire-x/Documents/Data_Skripsi_Programming/distance/Hasil-depan_detect_ball-distance_bismillah-tambah1.avi', cv2.VideoWriter_fourcc(*'MJPG'),15, size)
    # result_omni = cv2.VideoWriter('/home/fire-x/Documents/Data_Skripsi_Programming/distance/Hasil-omni_detect_ball-distance_bismillah-1.avi', cv2.VideoWriter_fourcc(*'MJPG'),15, size)

    while True:
      ret, frame = video_capture.read()
      success, frame_omni = video_capture_omni.read()
      if not ret:
          print('[INFO] Unable to read video / Video ended')
          break
    #   frame = cv2.flip(frame, 1) #omni cuy
    #   frame_omni = cv2.flip(frame_omni, 1)
      image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image_np_omni = cv2.cvtColor(frame_omni, cv2.COLOR_BGR2RGB)

      nframes = nframes + 1
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      # The model expects a batch of images, so also add an axis with `tf.newaxis`.
      input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
      input_tensor_omni = tf.convert_to_tensor(image_np_omni)[tf.newaxis, ...]

      # Pass frame through detector
      detections = detect_fn(input_tensor)
      detections_omni = detect_fn(input_tensor_omni)

      # Set detection parameters
      score_thresh = 0.22  # Minimum threshold for object detection
      max_detections = 1

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      scores = detections['detection_scores'][0, :max_detections].numpy()
      bboxes = detections['detection_boxes'][0, :max_detections].numpy()
      labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
      labels = [category_index[n]['name'] for n in labels]

      #ball detections
      scores_omni = detections_omni['detection_scores'][0, :max_detections].numpy()
      bboxes_omni = detections_omni['detection_boxes'][0, :max_detections].numpy()
      labels_omni = detections_omni['detection_classes'][0, :max_detections].numpy().astype(np.int64)
      labels_omni = [category_index[n]['name'] for n in labels_omni]

      # Display detections
      visualise_on_image(frame, bboxes, labels, scores, score_thresh)
          
      # Display detections kamera depan bola
      ball_detect(frame_omni, bboxes_omni, labels_omni,scores_omni, score_thresh)

      end_time = time.time()
      fps = int(1/(end_time - start_time))
      start_time = end_time

      # ukur performa per frame
      e2 = cv2.getTickCount()
      t = (e2 - e1)/cv2.getTickFrequency()
      t = t/nframes
    #   print('Rata-rata proses:' + str(t) + ' detik per frame')

      cv2.putText(frame, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
      cv2.putText(frame_omni, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
      cv2.imshow("Kamera Omnidirectional",frame)
      cv2.imshow("Kamera Depan", frame_omni)
      if cv2.waitKey(10) & 0xFF == ord('q'):
                # frame.release()
                cv2.destroyAllWindows()
                break
      
      #Write output video
    #   result.write(frame_omni)
    #   result_omni.write(frame)

    video_capture.release()