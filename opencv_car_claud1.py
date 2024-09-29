import cv2
import numpy as np
import time
from Adafruit_PCA9685 import PCA9685
from adafruit_servokit import ServoKit

# OpenCV 카메라 설정
cap = cv2.VideoCapture(0)

# I2C 설정
pwm = PCA9685(0x40, busnum=7)
servo_kit = ServoKit(channels=16, address=0x60)

class MotorDriver:
    def __init__(self):
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4
        self.MAX_SPEED = 100
        self.MIN_SPEED = 0

    def set_speed(self, motor_id, direction, speed):
        speed = max(min(speed, self.MAX_SPEED), self.MIN_SPEED)
        duty_cycle = int(speed * 40.95)

        if motor_id == 0:
            pwm.set_pwm(self.PWMA, 0, duty_cycle)
            pwm.set_pwm(self.AIN1, 0, 0 if direction == 'forward' else 4095)
            pwm.set_pwm(self.AIN2, 0, 4095 if direction == 'forward' else 0)
        else:
            pwm.set_pwm(self.PWMB, 0, duty_cycle)
            pwm.set_pwm(self.BIN1, 0, 0 if direction == 'forward' else 4095)
            pwm.set_pwm(self.BIN2, 0, 4095 if direction == 'forward' else 0)

    def stop_motor(self, motor_id):
        if motor_id == 0:
            pwm.set_pwm(self.PWMA, 0, 0)
        else:
            pwm.set_pwm(self.PWMB, 0, 0)

    def steer(self, angle):
        angle = max(min(angle, 180), 0)
        servo_kit.servo[0].angle = angle

motor = MotorDriver()

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([
        [(0, height), (width, height), (width // 2, height // 2 - 160)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def detect_lane(img):
    edges = process_image(img)
    cropped_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
    
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))
    
    left_avg = np.average(left_lines, axis=0) if len(left_lines) > 0 else None
    right_avg = np.average(right_lines, axis=0) if len(right_lines) > 0 else None
    
    return left_avg, right_avg

def control_vehicle(left_lane, right_lane):
    if left_lane is None and right_lane is None:
        return 90, 0  # 차선을 감지하지 못할 경우 정지
    
    if left_lane is not None and right_lane is not None:
        # 양쪽 차선이 모두 감지된 경우
        steering_angle = 90  # 직진
        speed = motor.MAX_SPEED
    elif left_lane is not None:
        # 왼쪽 차선만 감지된 경우
        steering_angle = 60  # 약간 우회전
        speed = motor.MAX_SPEED * 0.8
    elif right_lane is not None:
        # 오른쪽 차선만 감지된 경우
        steering_angle = 120  # 약간 좌회전
        speed = motor.MAX_SPEED * 0.8
    
    return steering_angle, speed

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 읽을 수 없습니다.")
            break

        left_lane, right_lane = detect_lane(frame)
        steering_angle, speed = control_vehicle(left_lane, right_lane)

        motor.steer(steering_angle)
        motor.set_speed(0, 'forward', speed)
        motor.set_speed(1, 'forward', speed)

        # 디버깅을 위한 화면 출력
        canny_image = process_image(frame)
        cropped_image = region_of_interest(canny_image)
        cv2.imshow("Canny Image", canny_image)
        cv2.imshow("Cropped Image", cropped_image)
        cv2.imshow("Original", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("프로그램 종료: 모터와 서보 모터를 정지 상태로 설정합니다.")

finally:
    motor.stop_motor(0)
    motor.stop_motor(1)
    motor.steer(90)
    cap.release()
    cv2.destroyAllWindows()