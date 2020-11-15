import os
import cv2
from base_camera import BaseCamera
import numpy as np
import tensorflow
import constant
import datetime
from imutils.video import FPS
from threading import Thread
import time
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.compiler.tensorrt import trt
from tensorflow.python.platform import gfile
import random
import string
import imutils

#pos = 0
#neg = 0


def get_random_alphanumeric_string():
    length = 20
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    return result_str

def read_pb_graph(model):
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


class ThreadedCamera(object):
    def __init__(self, source=0):

        self.capture = cv2.VideoCapture(source)
        time.sleep(2)
        self.thread = Thread(target=self.update, args=())

        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.capture.isOpened():
                self.capture.grab()
                time.sleep(0.005)

    def grab_frame(self):
        _, img = self.capture.retrieve()
        return img


def detect_and_predict_mask(frame, faceNet, maskNet, input, output, sess, set_num, face_num):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    t_start = datetime.datetime.now()
    faceNet.setInput(blob)
    detections = faceNet.forward()
    t_end = datetime.datetime.now()
    prediction_time = t_end - t_start
    print("PREDICTION TIME : ", (prediction_time.total_seconds()) * 1000, " ms")
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            if startX >= w or startY >= h:
                pass

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            
            try:
                face = cv2.resize(face, (224, 224))
                #cv2.imwrite(constant.BASEPATH + "/collect_data/{}/{}.jpg".format(set_num, face_num), face)
                #face_num += 1 
                
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))
            except:
                continue
            

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        for face in faces:
            preds.append(sess.run(output, feed_dict={input: face}))
        
       

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds), face_num


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        """
        Start Inference And Sent Frame One by one using 
        yield function.
        """
        fps = FPS()
        fps.start()
        current_fps = 0.0
        prediction_time = 0.0
        t_fps_start = datetime.datetime.now()
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        model = tensorflow.keras.models.load_model(
            constant.MODELFILELOCATION+'/'+constant.MODELSAVENAME)
        classList = []
        with tf.device('/gpu:0'):
            graph = tf.compat.v1.Graph()
            with graph.as_default():
                with tf.compat.v1.Session() as sess:
                    # read TensorRT model
                    frozen_graph = read_pb_graph(constant.MODELFILELOCATION +'/trt_face.pb')
                    tf.import_graph_def(frozen_graph, name='')
                    input = sess.graph.get_tensor_by_name(
                        'sequential_5_input:0')
                    output = sess.graph.get_tensor_by_name(
                        'sequential_7/dense_Dense4/Softmax:0')

                    with open(constant.MODELFILELOCATION+'/'+constant.CLASSTEXTSAVENAME, "r") as f:
                        for line in f:
                            classList.append(line.strip())
                    print("[INFO] loading face detector model...")
                    prototxtPath = constant.BASEPATH+"/face_detector/deploy.prototxt"
                    weightsPath = constant.BASEPATH+"/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
                    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
                    print("[ INFO ]:Video Streamming staring Now")
                    camera = ThreadedCamera(0)
                       # "rtsp://admin:Robro123@192.168.1.64:554/Streaming/Channels/101")
                    
                    random_string = get_random_alphanumeric_string()
                    face_num, set_num = 0, str(datetime.date.today())+'-'+random_string
                    #if not os.path.isfile(constant.BASEPATH + "/collect_data/{}".format(set_num)):
                    #    os.system("mkdir "+ constant.BASEPATH + "/collect_data/{}".format(set_num))
                    
                    with_mask, without_mask = 0, 0

                    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()
                    while True:
                        time.sleep(0.2)
                        # read current frame
                        img, break_flag = None, False
                        try:
                            img = camera.grab_frame()
                        except:
                            continue
                        if img is None:
                            print("[ INFO ]: Image is not there")
                            continue

                        frame = imutils.resize(img, width=224)

                        foregroundMask = backgroundSubtractor.apply(frame)

                        dilate = cv2.dilate(foregroundMask, None, iterations=1)
                        contours = cv2.findContours(
                            dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours = imutils.grab_contours(contours)

                        for contour in contours:
                            if cv2.contourArea(contour) > 2000:
                                break_flag = True
                                break
                        
                        if break_flag == True:
                            t_start = datetime.datetime.now()
                            (locs, preds), face_num = detect_and_predict_mask(
                                img, faceNet, model, input, output, sess, set_num, face_num)
                            
                            t_end = datetime.datetime.now()
                            prediction_time = t_end - t_start

                            fps.update()
                            if((t_end - t_fps_start).total_seconds() > 30.0):
                                fps.stop()
                                current_fps = fps.fps()
                                fps = FPS()
                                fps.start()
                                t_fps_start = datetime.datetime.now()

                            
                            labelFlag = True
                            if len(locs) == 0:
                                labelFlag = None
                            for (box, pred) in zip(locs, preds):
                                # unpack the bounding box and predictions
                                (startX, startY, endX, endY) = box
                                (mask, withoutMask) = pred[0]

                                # determine the class label and color we'll use to draw
                                # the bounding box and text
                                label = "Mask" if mask > withoutMask else "No Mask"
                                if label == "No Mask":
                                    labelFlag = False
                                    #cv2.imwrite(constant.BASEPATH + "/collect_data/without_mask/{}.jpg"
                                    #        .format(without_mask), img)
                                    without_mask += 1
                                color = (0, 255, 0) if label == "Mask" else (
                                    0, 0, 255)
                                
                                # include the probability in the label
                                label = "{}: {:.2f}%".format(
                                    label, max(mask, withoutMask) * 100)
                                
                                #accu = max(mask, withoutMask) * 10
                                #print("The label of the frame is : ", label)
                                # display the label and bounding box rectangle on the output
                                # frame
                                #if label == "No Mask":
                                #    cv2.imwrite(constant.BASEPATH + "/collect_data/with_mask/{}.jpg"
                                #            .format(with_mask), img)
                                #    without_mask += 1
                                
                                #else:
                                #    cv2.imwrite(constant.BASEPATH + "/collect_data/without_mask/{}.jpg"
                                #            .format(without_mask), img)
                                #    without_mask += 1
                                
                                cv2.putText(img, label, (startX, startY - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                                cv2.rectangle(img, (startX, startY),
                                              (endX, endY), color, 2)
                            
                            print("[ INFO ]:labelFlag:", labelFlag)
                            '''
                            if(labelFlag == True):
                                LED('ired', 'stableOff').start()
                                LED('igreen', 'stableOn').start()
                                #return pos, accu
                            elif(labelFlag == False):
                                LED('ired', 'stableOn').start()
                                LED('igreen', 'stableOff').start()
                                #return  neg, accu
                            else:
                                LED('ired', 'stableOff').start()
                                LED('igreen', 'stableOff').start()
                            '''
                        yield cv2.imencode('.jpg', img)[1].tobytes()
