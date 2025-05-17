import cv2
import os
path = "/mnt/e/Dataset/football/football_train/Match_1824_1_0_subclip_3/Match_1824_1_0_subclip_3.mp4"
print("Tồn tại:", os.path.exists(path))
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while cap.isOpened():
    flag, frame = cap.read()
    if flag:
        height, weight, _ = frame.shape
        # giam chieu can dua chieu nganh vao truoc
        cv2.imshow("demo", cv2.resize(frame, (weight//3 , height//3)))
    key = cv2.waitKey(20)
    if key == ord('q'):
        break

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model = LinearRegression
# from torch import nn
# import torch
#
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))