import cv2
from time import sleep, time
import threading


frames={}
scale = 0.8
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("./model/yolov4-tiny.weights", "./model/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

def readvid(i):
    vod = cv2.VideoCapture("./img/demo.mp4")
    while True:
        try:
            ret, frame = vod.read()
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            frames[i] = gpu_frame.download()
        except:
            pass

def showvid():
    while True:
        try:
            start = time()
            frame = frames[1]
            classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            for (classid, score, box) in zip(classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (class_names[classid[0]], score)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            end = time()

            fps_label = "FPS: %.2f " % (1 / (end - start))
            cv2.putText(frame, fps_label, (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('ss', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass


if __name__ == '__main__':
    load = threading.Thread(target=readvid,args=(1,))
    show = threading.Thread(target=showvid,args=())

    load.start()
    show.start()


