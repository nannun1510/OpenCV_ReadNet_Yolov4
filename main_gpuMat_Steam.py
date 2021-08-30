from main_cuda_thread import showvid
import cv2
from time import sleep, time
from flask import Flask, render_template, Response

app = Flask(__name__)

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

vod = cv2.VideoCapture("./img/demo.mp4")
def showdetect():
    while True:
            ret, frame = vod.read()
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            frame = gpu_frame.download()

            start = time()
            classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            for (classid, score, box) in zip(classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (class_names[classid[0]], score)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            end = time()

            fps_label = "FPS: %.2f " % (1 / (end - start))
            print(fps_label , end = "\r")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

#####################################################################

@app.route('/video_feed')
def video_feed():
    return Response(showdetect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

#####################################################################
if __name__ == '__main__':
    app.run(host='10.1.10.52', port=5000,debug=True)
