#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from ament_index_python.packages import get_package_share_directory
from collections import deque


class CollisionAINode(Node):

    def __init__(self):
        super().__init__('collision_ai_node')

        self.bridge = CvBridge()

        # frame sequence buffer
        self.frame_buffer = deque(maxlen=10)

        # ROS publisher
        self.pub = self.create_publisher(Bool, '/collision_warning', 10)

        # ROS subscriber
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10
        )

        MODEL_PATH = os.path.join(
            get_package_share_directory('collision_ai'),
            'best_collision_model.h5'
        )

        self.model = load_model(MODEL_PATH)

        self.SEQ_LEN = 10
        self.IMG_SIZE = 128

        # collision confirmation counter
        self.collision_count = 0
        self.COLLISION_CONFIRM = 3


    def callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        img = cv2.resize(frame, (self.IMG_SIZE, self.IMG_SIZE))
        img = img / 255.0

        self.frame_buffer.append(img)

        if len(self.frame_buffer) == self.SEQ_LEN:

            seq = np.array(self.frame_buffer)
            seq = seq.reshape(1, self.SEQ_LEN, self.IMG_SIZE, self.IMG_SIZE, 3)

            pred = self.model.predict(seq, verbose=0)
            pred_value = float(pred[0][0])

            # threshold improved
            if pred_value > 0.6:
                self.collision_count += 1
            else:
                self.collision_count = 0

            warning = self.collision_count >= self.COLLISION_CONFIRM

            self.pub.publish(Bool(data=warning))

            self.get_logger().info(
                f"Pred: {pred_value:.3f} | Count: {self.collision_count} | Warning: {warning}"
            )

            if warning:
                self.get_logger().warn("⚠ COLLISION RISK DETECTED")


def main(args=None):

    rclpy.init(args=args)

    node = CollisionAINode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()