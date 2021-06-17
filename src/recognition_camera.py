from mtcnn import MTCNN
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from model import CNN2, CNN3
from utils import index2emotion, cv2_img_add_text, results_to_zhuanzhu
import time

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=0, help="data source, 0 for camera 1 for video")
parser.add_argument("--video_path", type=str, default=None)
opt = parser.parse_args()
detector = MTCNN()

if opt.source == 1 and opt.video_path is not None:
    filename = opt.video_path
else:
    filename = None


def load_model():
    """
    加载本地模型
    :return:
    """
    model = CNN3()
    model.load_weights('./models/cnn3_best_weights.h5')
    return model


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """

    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img)
    resized_images.append(face_img[2:45, :])
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression():
    """
    实时预测
    :return:
    """
    # 参数设置
    model = load_model()

    border_color = (0, 0, 0)  # 黑框框
    font_color = (255, 255, 255)  # 白字字
    capture = cv2.VideoCapture(0)  # 指定0号摄像头
    if filename:
        capture = cv2.VideoCapture(filename)

    while True:
        # frame_time_start=time.time()
        _, frame = capture.read()  # 读取一帧视频，返回是否到达视频结尾的布尔值和这一帧的图像
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.flip(frame, 1)
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化


        frame_time_start=time.time()
        # MTCNN
        faces = detector.detect_faces(frame)
        # cascade = cv2.CascadeClassifier('./dataset/params/haarcascade_frontalface_alt.xml')  # 检测人脸
        # 利用分类器识别出哪个区域为人脸
        # faces = cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, minSize=(120, 120))
        # 如果检测到人脸
        if len(faces) > 0:
            for face in faces:
                face_time_start=time.time()
                x = face['box'][0]
                y = face['box'][1]
                w = face['box'][2]
                h = face['box'][3]
                face_img = cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY) # 脸部图片 # 灰度化
                faces = generate_faces(face_img)
                expression_results = model.predict(faces)
                # print(expression_results)
                result_sum = np.sum(expression_results, axis=0).reshape(-1)
                label_index = np.argmax(result_sum, axis=0)
                emotion = index2emotion(label_index)
                zhuanzhudu = results_to_zhuanzhu(expression_results)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)

                # 头部姿态，旋转向量
                
                frame = cv2_img_add_text(frame, emotion, x+10, y+30, font_color, 20)
                # frame = cv2_img_add_text(frame, str(results), x+30, y+20, font_color, 20)
                frame = cv2_img_add_text(frame, "专注度"+str(zhuanzhudu), x+10, y, font_color, 20)
                # puttext中文显示问题
                frame = cv2_img_add_text(frame, "人脸匹配度"+str(face['confidence']), x+10, y+60, font_color, 20)
                # cv2.putText(frame, emotion, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 4)

                face_time_end=time.time()
                print('face time cost',face_time_end-face_time_start,'s')

        cv2.imshow("expression recognition(press esc to exit)", frame)  # 利用人眼假象

        key = cv2.waitKey(30)  # 等待30ms，返回ASCII码

        # 如果输入esc则退出循环
        if key == 27:
            break
        frame_time_end=time.time()
        print('frame time cost',frame_time_end-frame_time_start,'s')
    capture.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 销毁窗口


if __name__ == '__main__':
    predict_expression()
