import os
import cv2
import time
import socket
import math
import numpy as np
import argparse

from Lib.StereoCamera_Lib import Stereo_Camera
from ocrtest import OCRModel
from Lib.DetectionLib import PaddleDetector
from Lib.color_dist import *
from Lib.Unitree_Lib import Unitree_Robot
from Lib.FaceLightLib import warning_light


class PatrolDog:
    def __init__(self):
        # 初始化socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ("127.0.0.1", 65432)

        # 初始化相机
        self.cap0 = Stereo_Camera(0)  # 前方
        self.cap0.camera_init()
        self.cap1 = Stereo_Camera(1)  # 下巴
        self.cap1.camera_init()

        # 初始化Aruco
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # 加载相机内参和畸变系数
        with np.load('camera_calibration.npz') as data:
            self.mtx = data['mtx']
            self.dist = data['dist']
            self.marker_length = 0.06

        # OCR 模型
        self.ocr_model = None

        # 图片检测模型
        self.detector = None

        # 机器狗动作
        self.unitree_robot = Unitree_Robot()

        self.middle_line_x = 335        # 下巴视野中线
        self.follow_path_angle = 7       # 路径跟随的左右偏移角度
        self.rectify_path_angle = 5      # 路径矫正的左右偏移角度
        self.follow_path_ruocuo = 20     # 路径跟随的左右偏移容错
        self.rectify_path_ruocuo = 5     # 路径矫正的左右偏移容错
        self.maxForwardSpeed = 1.0
        self.maxRotateSpeed = 0.9

        self.aruco_distance = 0.269

        # 初始化Flag
        self.animal = ['monkey', 'panda', 'wolf']
        self.Flag_animal = [0, 0, 0]    # 0:未检测到；1：检测到了；2：做动作了[monkey, panda, wolf][look_down, look_up, light]
        self.Flag_aruco = [2, 0, 0]     # 0:未检测到；1：检测到了；2：做动作了    [step, obstacle, destination]
        self.Flag_ocr = 0               # 0:未检测到ocr；1：检测到了ocr, 是否检测到ocr
        self.Flag_ocr_rectify = 0       # 0:未矫正；1：已矫正, ocr前的道路是否矫正
        self.Flag_llm_direction = 0     # 0:未判断；1：判断为left；2：判断为right, 大语言模型判断出的左右方向
        self.Flag_des_rectify = 0       # 0:未矫正；1：已矫正, 终点前的道路是否矫正
        self.is_right_angle = 0         # 0:未检测到直角；1：检测到直角了；
        self.Flag_rectify_direction = ""
        self.Flag_rectify_direction_ocr = ""

        self.show_save = [False, False]  # show:False, save:False (默认)
        self.out_folder_path = "video/PatrolDog"
        self.out0 = None
        self.out1 = None
        self.out2 = None

    def save_show_init(self):
        if self.show_save[1]:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_rate = 9.0  # Adjust the frame rate as needed
            frame_size = (640, 480)  # Adjust the frame size to match your camera's resolution
            if not os.path.exists(self.out_folder_path):
                os.makedirs(self.out_folder_path)
            self.out0 = cv2.VideoWriter(self.out_folder_path + '/' + 'forward.avi', fourcc, frame_rate, frame_size)
            self.out1 = cv2.VideoWriter(self.out_folder_path + '/' + 'chin.avi', fourcc, frame_rate, frame_size)
            self.out2 = cv2.VideoWriter(self.out_folder_path + '/' + 'forworad_chin.avi', fourcc, frame_rate, (640 * 2, 480))

    def waitForTask(self, task, duration):
        # task 为需要执行的任务
        # duration 为持续时间, 单位为秒
        start_time = time.time()
        while time.time() - start_time < duration:
            task()

    def calc_distance(self, p1, p2):
        return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2

    def calculate_angle(self, first_point, end_point):
        # 第一条线的向量
        A = (0, 480)

        # 第二条线的向量
        B = (end_point[0] - first_point[0], end_point[1] - first_point[1])

        # 计算点积
        dot_product = A[0] * B[0] + A[1] * B[1]

        # 计算向量的模
        magnitude_A = math.sqrt(A[0] ** 2 + A[1] ** 2)
        magnitude_B = math.sqrt(B[0] ** 2 + B[1] ** 2)

        # 计算夹角的余弦值
        cos_theta = dot_product / (magnitude_A * magnitude_B)

        # 计算夹角（弧度）
        theta_radians = math.acos(cos_theta)

        # 转换为角度
        theta_degrees = math.degrees(theta_radians)

        if theta_degrees > 90:
            theta_degrees = 180 - theta_degrees

        return theta_degrees, A[0] * B[1] - A[1] * B[0] > 0

    def calculate_rotatespeed(self, angle, min_speed=0.0, max_speed=0.9, k=0.1):
        angle = max(0, min(angle, 90))
        rotatespeed = min_speed + (max_speed - min_speed) * (1 - np.exp(-k * angle))
        return max(0.12, rotatespeed)

    def follow_path(self, image1, color):
        # HSV
        color_lower = np.array(color_dist[color]["Lower"], np.uint8)
        color_upper = np.array(color_dist[color]["Upper"], np.uint8)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 矩形结构
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 矩形结构

        # 图像处理
        hsvFrame = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsvFrame, color_lower, color_upper)
        color_mask = cv2.medianBlur(color_mask, 9)  # 中值滤波
        color_mask = cv2.erode(color_mask, erode_kernel)  # 腐蚀
        color_mask = cv2.dilate(color_mask, dilate_kernel)  # 膨胀
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        color_image = image1.copy()
        AREA_THRESHOLD = 6000
        mask_roi = color_mask[:300, :]  # 划定ROI区域

        contours_roi, _ = cv2.findContours(mask_roi[:, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_roi = list(filter(lambda x: cv2.contourArea(x) > AREA_THRESHOLD, contours_roi))

        if len(contours_roi):
            c = max(contours_roi, key=cv2.contourArea)
            # 计算轮廓的重心
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))

                if self.calc_distance(box[0], box[1]) > self.calc_distance(box[1], box[2]):
                    pt1 = (box[1] + box[2]) // 2
                    pt2 = (box[3] + box[0]) // 2
                else:
                    pt1 = (box[0] + box[1]) // 2
                    pt2 = (box[2] + box[3]) // 2

                # pt1, pt2 为与长边平行中线的端点
                # 可以基于此计算狗轴线与路径中线的角度
                if abs(pt1[1]-pt2[1]) <= 50:
                    if abs(pt1[0] - self.middle_line_x) < abs(pt2[0] - self.middle_line_x):
                        pt1, pt2 = pt2, pt1
                elif pt1[1] > pt2[1]:
                    pt1, pt2 = pt2, pt1

                pt1_, pt2_ = pt1.copy(), pt2.copy()
                pt1_[0] -= 320
                pt2_[0] -= 320

                angle, is_right = self.calculate_angle(pt1, pt2)

                forwardSpeed = 0.30
                rotateSpeed = 0.05
                sidewaySpeed = 0.0

                if angle >= self.follow_path_angle:
                    # if self.Flag_animal.count(2) == 2:
                    #     rotateSpeed = 0.9
                    #     forwardSpeed = 0.15
                    # else:
                    #     rotateSpeed = 0.12
                    #     forwardSpeed = 0.12

                    rotateSpeed = self.calculate_rotatespeed(angle)
                    if self.Flag_animal.count(2) == 2:
                        forwardSpeed = 0.2
                    else:
                        forwardSpeed = 0.12

                    if is_right:
                        direction = "F_right"
                        rotateSpeed = -rotateSpeed
                    else:
                        direction = "F_left"
                        rotateSpeed = rotateSpeed
                else:
                    if cx < self.middle_line_x - self.follow_path_ruocuo:
                        direction = "F_left_p"
                        if self.Flag_animal.count(2) == 2:
                            forwardSpeed = 0.3
                            # rotateSpeed = 0.4
                            sidewaySpeed = 0.12
                        else:
                            forwardSpeed = 0.12
                            sidewaySpeed = 0.12

                    elif cx > self.middle_line_x + self.follow_path_ruocuo:
                        direction = "F_right_p"
                        if self.Flag_animal.count(2) == 2:
                            forwardSpeed = 0.3
                            # rotateSpeed = -0.4
                            sidewaySpeed = -0.12
                        else:
                            forwardSpeed = 0.12
                            sidewaySpeed = -0.12

                    else:
                        direction = "F_forward"
                        if self.Flag_animal.count(2) == 2 and self.Flag_animal.count(0) == 1:
                            forwardSpeed = 0.5

                if self.Flag_animal.count(1) == 1 and angle >= 85:
                    self.is_right_angle = 1
                    self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0,
                                                     sidewaySpeed=0.0,
                                                     rotateSpeed=0, speedLevel=0,
                                                     bodyHeight=0.0)
                else:
                    self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=forwardSpeed,
                                                     sidewaySpeed=sidewaySpeed,
                                                     rotateSpeed=rotateSpeed, speedLevel=0,
                                                     bodyHeight=0.0)

                print("direction:", direction, "; angle:", f"{angle:.2f}", "; forward:", forwardSpeed, "; rotate:",
                      rotateSpeed, "; sideway:", sidewaySpeed)

                if self.show_save[1]:
                    cv2.drawContours(color_image, [box], 0, (0, 0, 255), 2)
                    cv2.drawContours(color_image, [c], -1, (0, 255, 0), 2)
                    cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.line(color_image, tuple(pt1), tuple(pt2), (255, 0, 0), 2)
                    cv2.line(color_image, (self.middle_line_x, 0), (self.middle_line_x, 480), (0, 0, 255), 2)
                    cv2.line(color_image, (self.middle_line_x - self.follow_path_ruocuo, 0),
                             (self.middle_line_x - self.follow_path_ruocuo, 480), (0, 0, 255), 1)
                    cv2.line(color_image, (self.middle_line_x + self.follow_path_ruocuo, 0),
                             (self.middle_line_x + self.follow_path_ruocuo, 480), (0, 0, 255), 1)
                    # 写入方向
                    cv2.putText(color_image, direction + " " + "{:.3f}".format(angle), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255), 2,
                                cv2.LINE_AA)

        else:
            print("Not find path")
            self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=-0.12,
                                             sidewaySpeed=0,
                                             rotateSpeed=0.06, speedLevel=0, bodyHeight=0.0)

        return color_image

    def rectify_path(self, image1, color):
        color_lower = np.array(color_dist[color]["Lower"], np.uint8)
        color_upper = np.array(color_dist[color]["Upper"], np.uint8)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 矩形结构
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 矩形结构

        # 图像处理
        hsvFrame = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsvFrame, color_lower, color_upper)
        color_mask = cv2.medianBlur(color_mask, 9)  # 中值滤波
        color_mask = cv2.erode(color_mask, erode_kernel)  # 腐蚀
        color_mask = cv2.dilate(color_mask, dilate_kernel)  # 膨胀
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        color_image = image1.copy()
        AREA_THRESHOLD = 200
        color_mask[0:100, :] = 0
        color_mask[300:, :] = 0
        mask_roi = color_mask[:, :]  # 划定ROI区域

        contours_roi, _ = cv2.findContours(mask_roi[:, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_roi = list(filter(lambda x: cv2.contourArea(x) > AREA_THRESHOLD, contours_roi))

        # 假设最大轮廓是我们的线条
        if len(contours_roi):
            c = max(contours_roi, key=cv2.contourArea)

            # 计算轮廓的重心
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))

                if self.calc_distance(box[0], box[1]) > self.calc_distance(box[1], box[2]):
                    pt1 = (box[1] + box[2]) // 2
                    pt2 = (box[3] + box[0]) // 2
                else:
                    pt1 = (box[0] + box[1]) // 2
                    pt2 = (box[2] + box[3]) // 2

                # pt1, pt2 为与长边平行中线的端点
                # 可以基于此计算狗轴线与路径中线的角度
                if pt1[1] > pt2[1]:
                    pt1, pt2 = pt2, pt1

                pt1_, pt2_ = pt1.copy(), pt2.copy()
                pt1_[0] -= 320
                pt2_[0] -= 320

                angle, is_right = self.calculate_angle(pt1_, pt2_)     # 起点,终点
                sidewaySpeed = 0.0
                rotateSpeed = 0.05
                sleep_time = 0.0
                # distance = abs(self.middle_line_x - cx)

                if angle >= self.rectify_path_angle:
                    rotateSpeed = self.calculate_rotatespeed(angle)

                    if is_right:
                        direction = "R_right"
                        rotateSpeed = -rotateSpeed

                    else:
                        direction = "R_left"
                        rotateSpeed = rotateSpeed

                    self.Flag_rectify_direction = direction

                else:
                    if cx < self.middle_line_x - self.rectify_path_ruocuo:
                        direction = "R_left_p"
                        sidewaySpeed = 0.12
                        self.Flag_rectify_direction = direction

                    elif cx > self.middle_line_x + self.rectify_path_ruocuo:
                        direction = "R_right_p"
                        sidewaySpeed = -0.12
                        self.Flag_rectify_direction = direction

                    else:
                        direction = "R_forward"
                        self.Flag_ocr_rectify = 1
                        self.Flag_des_rectify = 0
                        if self.Flag_rectify_direction == "R_right_p":
                            sidewaySpeed = 0.12
                            sleep_time = 0.3
                        elif self.Flag_rectify_direction == "R_left_p":
                            sidewaySpeed = -0.12
                            sleep_time = 0.3
                        self.Flag_rectify_direction = ""

                self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.01, sidewaySpeed=sidewaySpeed,
                                                 rotateSpeed=rotateSpeed, speedLevel=0, bodyHeight=0)

                if direction == "R_forward":
                    time.sleep(sleep_time)
                    if self.Flag_llm_direction == 0:
                        time.sleep(0.2)
                        self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0,
                                                         rotateSpeed=0, speedLevel=0, bodyHeight=0)
                        time.sleep(0.1)

                print("direction:", direction, "; angle:", f"{angle:.2f}", "; forward:0.01", "; rotate:", rotateSpeed,
                      "; sideway:", sidewaySpeed)

                if self.show_save[1]:
                    cv2.drawContours(color_image, [box], 0, (0, 0, 255), 2)
                    cv2.drawContours(color_image, [c], -1, (0, 255, 0), 2)
                    cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.line(color_image, tuple(pt1), tuple(pt2), (255, 0, 0), 2)
                    cv2.line(color_image, (self.middle_line_x, 0), (self.middle_line_x, 480), (0, 0, 255), 2)
                    cv2.line(color_image, (self.middle_line_x - self.rectify_path_ruocuo, 0), (self.middle_line_x - self.rectify_path_ruocuo, 480), (0, 0, 255), 1)
                    cv2.line(color_image, (self.middle_line_x + self.rectify_path_ruocuo, 0), (self.middle_line_x + self.rectify_path_ruocuo, 480), (0, 0, 255), 1)
                    # 写入方向
                    if is_right:
                        cv2.putText(color_image, direction + " -" + "{:.3f}".format(angle), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255), 2,
                                    cv2.LINE_AA)
                    else:
                        cv2.putText(color_image, direction + " +" + "{:.3f}".format(angle), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255), 2,
                                    cv2.LINE_AA)

        return color_image

    def animal_detection(self, image0):
        result = self.detector.predict_image(image0)

        # if self.show_save:
        #     image0 = self.detector.visualize_boxes(image0, result)

        np_boxes = result['boxes']
        if np_boxes[:, 1] > 0.5:
            for idx in range(np_boxes.shape[0]):
                cls_id = int(np_boxes[idx, 0])

                print(self.animal[cls_id], np_boxes[:, 1])

                if cls_id == 0 and self.Flag_animal[0] == 0:
                    self.Flag_animal[0] = 1

                elif cls_id == 1 and self.Flag_animal[1] == 0:
                    self.Flag_animal[1] = 1

                elif cls_id == 2 and self.Flag_animal[2] == 0:
                    self.Flag_animal[2] = 1

    def aruco_detection(self, image0, image1):
        # aruco id2
        if self.Flag_aruco[1] == 0:
            gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            corners0, ids0, rejectedImgPoints0 = cv2.aruco.detectMarkers(gray0, self.aruco_dict, parameters=self.parameters)
    
            if ids0 is not None:
                if ids0[0] == 2:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners0, self.marker_length, self.mtx, self.dist)
                    for rvec, tvec in zip(rvecs, tvecs):
                        # 获取相机与标记的距离（单位：米）
                        distance = np.linalg.norm(tvec)
                        print("aruco id2:", distance)
                        if distance <= self.aruco_distance:
                            self.Flag_aruco[1] = 1
        # aruco id3
        elif self.Flag_aruco[2] == 0:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            corners1, ids1, rejectedImgPoints1 = cv2.aruco.detectMarkers(gray1, self.aruco_dict, parameters=self.parameters)
            if ids1 is not None:
                if ids1[0] == 3:
                    self.Flag_aruco[2] = 1

    def ocr_detection(self, image0):
        res_str = ""
        dt_boxes, rec_res = self.ocr_model(image0)

        if len(rec_res) != 0:
            for text, score in rec_res:
                # print(text, score)
                # if score >= 0.8:
                res_str += text

        print("OCR_det_content:", res_str)
        return res_str

    def start_point_action(self):
        # 左前
        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.50, sidewaySpeed=0.50,
                                                 rotateSpeed=0.05, speedLevel=0, bodyHeight=0),
            1.4
        )

    def animal_action(self, animal_kind=-1):
        if animal_kind == 0:
            print("monkey_action")
            self.waitForTask(
                lambda: self.unitree_robot.robot_pose(roll=0.0, pitch=1.0, yaw=0.0, bodyHeight=0.0),
                1.5
            )
            time.sleep(1.5)
            self.Flag_animal[0] = 2

        elif animal_kind == 1:
            print("panda_action")
            self.waitForTask(
                lambda: self.unitree_robot.robot_pose(roll=0.0, pitch=-1.0, yaw=0.0, bodyHeight=0.0),
                1.5
            )
            time.sleep(1.5)
            self.Flag_animal[1] = 2

        elif animal_kind == 2:
            print("wolf_action")
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0,
                                                         rotateSpeed=0.0, speedLevel=0, bodyHeight=0),
                0.1
            )
            warning_light()
            self.Flag_animal[2] = 2
        
    def animal_next_action(self):
        # 第一个动物动作做完后的写死动作，左后
        if self.Flag_animal.count(2) == 1:
            # 左后
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=-0.27, sidewaySpeed=0.50,
                                                         rotateSpeed=0.05, speedLevel=0, bodyHeight=0),
                2.5
            )
            # 发零
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.00, sidewaySpeed=0.00,
                                                         rotateSpeed=0.05, speedLevel=0, bodyHeight=0),
                0.1
            )

        # 第二个动物动作做完后直接过台阶
        elif self.Flag_animal.count(2) == 2:
            self.step_action()

        # 第三个动物动作做完后， 左前
        elif self.Flag_animal.count(2) == 3:
            # 左前
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.08, sidewaySpeed=0.05,
                                                         rotateSpeed=-0.95, speedLevel=0, bodyHeight=0),
                2.0
            )
            # 发零
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.00, sidewaySpeed=0.00,
                                                         rotateSpeed=0.06, speedLevel=0, bodyHeight=0),
                0.1
            )

    def ocr_action(self):
        if self.Flag_llm_direction == 2:
            self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0,
                                             sidewaySpeed=-0.35,
                                             rotateSpeed=0.05, speedLevel=0,
                                             bodyHeight=0)
            time.sleep(1.2)
            self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0,
                                        sidewaySpeed=-0.23,
                                        rotateSpeed=0.05, speedLevel=0,
                                        bodyHeight=0)
            time.sleep(1.5)
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.3,
                                                    sidewaySpeed=0.0,
                                                    rotateSpeed=0.05, speedLevel=0,
                                                    bodyHeight=0),
                2.0
            )
            self.Flag_ocr_rectify = 0

        # left
        elif self.Flag_llm_direction == 1:
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.20,
                                                    sidewaySpeed=0.0,
                                                    rotateSpeed=0.5, speedLevel=0,
                                                    bodyHeight=0),
                2.0
            )
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0,
                                                    sidewaySpeed=0.0,
                                                    rotateSpeed=0.00, speedLevel=0,
                                                    bodyHeight=0),
                0.1
            )
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.21,
                                                    sidewaySpeed=0,
                                                    rotateSpeed=0.06, speedLevel=0,
                                                    bodyHeight=0),
                2.0
            )
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0,
                                                    sidewaySpeed=0.0,
                                                    rotateSpeed=0.00, speedLevel=0,
                                                    bodyHeight=0),
                0.1
            )
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0,
                                                         sidewaySpeed=-0.03,
                                                         rotateSpeed=-0.40, speedLevel=0,
                                                         bodyHeight=0),
                2.0
            )
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0,
                                                         sidewaySpeed=0.0,
                                                         rotateSpeed=0.00, speedLevel=0,
                                                         bodyHeight=0),
                0.1
            )
            self.Flag_ocr_rectify = 0

    def obstacle_action(self):
        # right
        if self.Flag_llm_direction == 2:
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.02, sidewaySpeed=0.0,
                                                         rotateSpeed=0.55, speedLevel=0, bodyHeight=1.0),
                2
            )
            # forward， sidewaySpeed=0.01 改为0.00
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.21, sidewaySpeed=0.00,
                                                         rotateSpeed=0.06, speedLevel=0, bodyHeight=1.0),
                1.0
            )
            # forward
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.20, sidewaySpeed=0.00,
                                                         rotateSpeed=0.06, speedLevel=0, bodyHeight=1.0),
                1.5
            )
            # 往前右旋
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.1, sidewaySpeed=0.0,
                                                         rotateSpeed=-0.45, speedLevel=0, bodyHeight=1.0),
                2
            )
            # 向前
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.50, sidewaySpeed=0.00,
                                                         rotateSpeed=0.05, speedLevel=0, bodyHeight=0.0),
                0.8
            )
            self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.12, sidewaySpeed=0.00,
                                        rotateSpeed=0.05, speedLevel=0, bodyHeight=0)
            time.sleep(1.2)

        # left
        elif self.Flag_llm_direction == 1:
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.05, sidewaySpeed=-0.01,
                                                         rotateSpeed=-0.50, speedLevel=0, bodyHeight=1.0),
                2
            )
            # forward
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.18, sidewaySpeed=0.00,
                                                         rotateSpeed=0.06, speedLevel=0, bodyHeight=1.0),
                1.8
            )
            # 往前左旋
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.2, sidewaySpeed=0.02,
                                                         rotateSpeed=0.60, speedLevel=0, bodyHeight=1.0),
                2.0
            )
            # 向前
            self.waitForTask(
                lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.50, sidewaySpeed=0.00,
                                                         rotateSpeed=0.05, speedLevel=0, bodyHeight=0),
                0.6
            )
            self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.12, sidewaySpeed=0.00,
                                             rotateSpeed=0.05, speedLevel=0, bodyHeight=0)
            time.sleep(1.2)

        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0,
                                                     sidewaySpeed=0.0,
                                                     rotateSpeed=0.00, speedLevel=0,
                                                     bodyHeight=0),
            0.1
        )

    def step_action(self):
        print("step")
        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0, sidewaySpeed=0.0,
                                                     rotateSpeed=1.00, speedLevel=0, bodyHeight=0),
            2.1
        )

        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=-0.48,
                                                     rotateSpeed=0.06, speedLevel=0, bodyHeight=0),
            1.0
        )

        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.12, sidewaySpeed=0.0,
                                                     rotateSpeed=0.02, speedLevel=0, bodyHeight=0),
            0.1
        )

        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=3, forwardSpeed=0.5, sidewaySpeed=0.00,
                                                     rotateSpeed=0.00, speedLevel=0, bodyHeight=0),
            4.6
        )
        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.00, sidewaySpeed=0.00,
                                                     rotateSpeed=0.0, speedLevel=0, bodyHeight=0),
            0.1
        )

    def des_action(self):
        print("destination")
        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0,
                                                     rotateSpeed=-0.98, speedLevel=0, bodyHeight=0),
            2.0
        )
        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.01, sidewaySpeed=0.55,
                                                     rotateSpeed=0.06, speedLevel=0, bodyHeight=0),
            0.65
        )
        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.12, sidewaySpeed=0.0,
                                                     rotateSpeed=0.06, speedLevel=0, bodyHeight=0),
            0.15
        )
        self.waitForTask(
            lambda: self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.0, sidewaySpeed=0.0,
                                                     rotateSpeed=0.00, speedLevel=0, bodyHeight=0),
            0.8
        )
        self.waitForTask(
            lambda: self.unitree_robot.robot_pose(roll=0.0, pitch=1.0, yaw=0.0, bodyHeight=1.0),
            5
        )

    def main_control(self, res_str):
        # 做相应的动物动作以及后续动作
        if self.Flag_animal.count(1) == 1:
            if self.is_right_angle == 1:
                cls_id = next((index for index, value in enumerate(self.Flag_animal) if value == 1), None)
                self.animal_action(cls_id)
                self.animal_next_action()
                self.is_right_angle = 0
            else:
                return False
            
        # OCR识别
        elif res_str != "":
            # 先进行道路矫正，道路正则开始大语言模型判断
            # if self.Flag_ocr_rectify == 1:
            print("ocr_rectify_end!")

            if args.turn == 1:
                # 左转
                time.sleep(1.0)
                self.Flag_llm_direction = 1
                print("ocr_action:left")

            elif args.turn == 2:
                # 右转
                time.sleep(1.0)
                self.Flag_llm_direction = 2
                print("ocr_action:right")

            else:
                # 正常 OCR + 大语言

                self.s.sendto(res_str.encode(), self.addr)
                data, addr = self.s.recvfrom(1024)
                print("LLM_rec_content:" + data.decode())
                if data.decode() == "left":
                    self.Flag_llm_direction = 1
                    print("ocr_action:left")
                elif data.decode() == "right":
                    self.Flag_llm_direction = 2
                    print("ocr_action:right")
                else:
                    print("ocr_action:wrong")
                    self.Flag_llm_direction = 1    # 默认为right

            self.ocr_action()

        # Aruco识别
        elif self.Flag_aruco.count(1) == 1:
            if self.Flag_aruco[1] == 1:
                self.Flag_aruco[1] = 2
                self.obstacle_action()
                self.Flag_des_rectify = 1

            elif self.Flag_aruco[2] == 1:
                self.Flag_aruco[2] = 2
                self.des_action()

        else:
            return False

        return True

    def start(self):
        print("Start!")
        start_time = time.time()
        self.start_point_action()

        while True:
            image0 = self.cap0.rgb_image()
            image1 = self.cap1.rgb_image()

            new_image0 = image0
            new_image1 = image1

            res_str = ""

            # 道路矫正
            if (self.Flag_animal.count(2) == 3 and self.Flag_ocr_rectify == 0) or self.Flag_des_rectify == 1:
                new_image1 = self.rectify_path(image1, color="chin_black_path")

            else:
                # 动物识别
                if self.Flag_animal.count(2) < 3:
                    if self.Flag_animal.count(1) == 0:
                        self.animal_detection(image0)
                        
                # OCR识别
                elif self.Flag_llm_direction == 0:
                    if args.turn == 1 or args.turn == 2:
                        res_str = str(args.turn)
                    else:
                        res_str = self.ocr_detection(image0)

                # Aruco识别
                elif self.Flag_aruco.count(2) < 3:
                    self.aruco_detection(image0, image1)

                # 主控/巡线
                if not self.main_control(res_str):
                    # 道路已矫正，直接直行
                    if self.Flag_ocr_rectify == 1 and self.Flag_des_rectify == 0:
                        self.unitree_robot.robot_walking(gaitType=1, forwardSpeed=0.12,
                                                         sidewaySpeed=0,
                                                         rotateSpeed=0.06, speedLevel=0,
                                                         bodyHeight=0)
                    else:
                        new_image1 = self.follow_path(image1, color="chin_black_path")

            image = cv2.hconcat([new_image0, new_image1])  # 水平拼接

            if self.show_save[0]:
                cv2.imshow("image", image)

                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    break

            if self.show_save[1]:
                self.out0.write(new_image0)
                self.out1.write(new_image1)
                self.out2.write(image)

            # 结束
            if self.Flag_aruco[2] == 2:
                break

        print("total_time:", time.time()-start_time)
        self.end()

    def end(self):
        self.cap0.release()
        self.cap1.release()

        if self.show_save[1]:
            self.out0.release()
            self.out1.release()
            self.out2.release()

        print("End!")


def main():
    dog = PatrolDog()
    dog.detector = PaddleDetector("./model/ppyolo_tiny_650e_coco_1", device='GPU')
    dog.ocr_model = OCRModel()

    dog.maxForwardSpeed = 0.3
    dog.maxRotateSpeed = 0.9
    dog.follow_path_angle = 5
    dog.rectify_path_angle = 1
    dog.follow_path_ruocuo = 20  
    dog.rectify_path_ruocuo = 5  
    dog.middle_line_x = 335

    dog.show_save = [False, True]
    dog.out_folder_path = "video/PatrolDog"
    dog.save_show_init()
    print("Init over!")
    dog.start()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Patrol dog program.")
    parse.add_argument("-t", "--turn",
                       type=int,
                       default=-1)
    args = parse.parse_args()
    main()
