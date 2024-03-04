import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2


sio = socketio.Server()

app = Flask(__name__)


def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


# connect simulator with python code
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    # as sooon as the connection establishes
    # set steer and throttle to 0 such that car starts at stationary
    send_control(0, 0)

# predict from the model and keep on sending to the simulator


def send_control(steering_angle, throttle):
    sio.emit('steer', data={'steering_angle': steering_angle.__str__(),
                            'throttle': throttle.__str__()})


# simulator will send back data which contains the current
# image of present location of car
@sio.on('telemetry')
def telemetry(sid, data):
    speed_limit = 10
    speed = float(data['speed'])
    # data['image'] is base64 encoded
    image = Image.open(BytesIO(base64.b64decode(data['image'])))

    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])

    steering_angle = float(model.predict(image))
    throttle = 1.0 - (speed/speed_limit)
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


if __name__ == '__main__':

    # import tensorflow as tf
    # print("Num GPUs Available: ", len(
    #     tf.config.experimental.list_physical_devices('GPU')))

    model = load_model('model2.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


# simp_model = model with no augmentation image
# model1 = model with augmented image and 4 layers of dropout
    # works good for validation data like overfitting
    # not much good for training data
# model2 = model with augmented image but no dropout layers
    # works very good for validation data
    # same condition = not much good for training data
# model3= model with augmented image no dropout layer but decreased learningrate=0.0001, Adam
