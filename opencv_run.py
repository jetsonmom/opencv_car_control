import cv2
import numpy as np
import time
from Adafruit_PCA9685 import PCA9685
from adafruit_servokit import ServoKit

# OpenCV 카메라 설정
cap = cv2.VideoCapture(0)

# I2C 설정
pwm = PCA9685(0x40)
pwm.set_pwm_freq(50)
servo_kit = ServoKit(channels=16)

class MotorDriver:
    def __init__(self):
        self.Dir = ['forward', 'backward']

    def set_speed(self, motor_id, direction, speed):
        if speed > 100:
            speed = 100  # 최대 속도 제한
        duty_cycle = int(speed * 40.95)
        if direction == 'forward':
            pwm.set_pwm(motor_id * 2, 0, 0)
            pwm.set_pwm(motor_id * 2 + 1, 0, duty_cycle)
        else:
            pwm.set_pwm(motor_id * 2, 0, duty_cycle)
            pwm.set_pwm(motor_id * 2 + 1, 0, 0)

    def stop_motor(self, motor_id):
        pwm.set_pwm(motor_id * 2, 0, 0)
        pwm.set_pwm(motor_id * 2 + 1, 0, 0)

    def steer(self, angle):
        servo_kit.servo[0].angle = angle

motor = MotorDriver()

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img

try:
    while True:
        ret, frame = cap.read()
        if ret:
            canny_image = process_image(frame)
            cropped_image = region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
            line_image = display_lines(frame, lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            cv2.imshow("result", combo_image)

            if lines is not None:
                x1, y1, x2, y2 = lines[0][0]
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                if slope < 0:
                    motor.steer(45)
                else:
                    motor.steer(135)
            else:
                motor.steer(90)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("프로그램 종료: 모터와 서보 모터를 정지 상태로 설정합니다.")
    motor.stop_motor(0)  # 모터 정지
    motor.stop_motor(1)  # 모터 정지
    motor.steer(90)      # 서보 모터를 90도로 조정하여 중립 위치로 설정

finally:
    cap.release()
    cv2.destroyAllWindows()
