import numpy as np
import cv2

############# Go 1 Camera list ################
    #  used
    # cam_id nano_id dev_id   port_id   位置
    #   0      13       1      9201     前方
    #   1      13       0      9202     下巴
    #   2      14       0      9203     左方
    #   3      14       1      9204     右方
    #   4      15       0      9205     腹部（默认）

###############################################

class Stereo_Camera:
    def __init__(self, camera_id=None, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_id = camera_id

        self.cap_dict = {}

    def camera_init(self):
        '''
        摄像头初始化
        '''
        IpLastSegment = 15
        udpPORT = [9201, 9202, 9203, 9204, 9205] # 端口：下巴，前方，左，右，腹部
        udpstrPrevData = "udpsrc address=192.168.123."+ str(IpLastSegment) + " port="
        udpstrBehindData = " ! application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink";
        if self.camera_id == None:
            for port, id in enumerate(udpPORT):
                udpSendIntegratedPipe = udpstrPrevData +  str(port) + udpstrBehindData
                cap = cv2.VideoCapture(udpSendIntegratedPipe, cv2.CAP_GSTREAMER)    # cv2.CAP_GSTREAMER  cv2.CAP_V4L2
                self.cap_dict.update({id: cap})

        elif type(self.camera_id) == list:
            for cam_id, id in enumerate(self.camera_id):
                udpSendIntegratedPipe = udpstrPrevData +  str(udpPORT[cam_id]) + udpstrBehindData
                cap = cv2.VideoCapture(udpSendIntegratedPipe, cv2.CAP_GSTREAMER)
                self.cap_dict.update({cam_id: cap})
                print("camera {} has been started sucessfully!".format(cam_id))
        elif type(self.camera_id) == int:
            udpSendIntegratedPipe = udpstrPrevData +  str(udpPORT[self.camera_id]) + udpstrBehindData
            cap = cv2.VideoCapture(udpSendIntegratedPipe, cv2.CAP_GSTREAMER)
            self.cap_dict.update({self.camera_id: cap})
        print("---------------------------------------------",\
              "\ncamera list:", self.cap_dict, "\n", \
              "---------------------------------------------")


    def rgb_image(self, cam_id=None):
        if cam_id == None:
            cam_id = self.camera_id
        _, rgb_image = self.cap_dict[cam_id].read()
        rgb_image = cv2.resize(rgb_image, (640, 480))
        if cam_id == 0:
            rgb_image = cv2.flip(rgb_image, -1) # 垂直+水平翻转
        elif cam_id == 1:
            rgb_image = cv2.flip(rgb_image, -1) # 垂直+水平翻转
            pass
        return rgb_image

    def forward_image(self):
        _, rgb_image = self.cap_dict[0].read()
        rgb_image = cv2.resize(rgb_image, (640, 480))
        rgb_image = cv2.flip(rgb_image, -1) # 垂直+水平翻转
        return rgb_image

    def chin_image(self):
        _, rgb_image = self.cap_dict[1].read()
        rgb_image = cv2.resize(rgb_image, (640, 480))
        rgb_image = cv2.flip(rgb_image, -1) # 垂直+水平翻转
        return rgb_image

    # def left_image(self):
        # if len(self.cap_list) > 2:

    def release(self):
        for cap in self.cap_dict.values():
            cap.release()


# cam = Stereo_Camera(camera_id=[0,1])
# cam.camera_init()
# while True:
#     forward_img = cam.forward_image()
#     chin_img = cam.chin_image()
#     cv2.imshow("forward_img", forward_img)
#     cv2.imshow("chin_img", chin_img)
#     if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
#         cv2.destroyAllWindows()
#         break
