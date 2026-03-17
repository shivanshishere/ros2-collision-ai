#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FakeCameraNode(Node):
    def __init__(self):
        super().__init__('fake_camera_node')
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # Absolute path to your test video
        self.cap = cv2.VideoCapture('/mnt/c/Users/Shivansh/Downloads/istockphoto-948688042-640_adpp_is.mp4')
        
        if not self.cap.isOpened():
            self.get_logger().error("Video file cannot be opened!")
        
        self.timer = self.create_timer(1/30, self.timer_callback)  # 30 FPS
        self.frame_count = 0

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub.publish(msg)
        self.frame_count += 1
        self.get_logger().info(f"Published frame #{self.frame_count}")

def main(args=None):
    rclpy.init(args=args)
    node = FakeCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()