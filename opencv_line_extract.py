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

    def set_speed(self, motor_id, direction, speed):
        if speed > 100:
            speed = 100  # 최대 속도 제한
        duty_cycle = int(speed * 40.95)  # 0-100 범위를 0-4095로 변환

        if motor_id == 0:
            pwm.set_pwm(self.PWMA, 0, duty_cycle)
            if direction == 'forward':
                pwm.set_pwm(self.AIN1, 0, 0)
                pwm.set_pwm(self.AIN2, 0, 4095)
            else:
                pwm.set_pwm(self.AIN1, 0, 4095)
                pwm.set_pwm(self.AIN2, 0, 0)
        else:
            pwm.set_pwm(self.PWMB, 0, duty_cycle)
            if direction == 'forward':
                pwm.set_pwm(self.BIN1, 0, 0)
                pwm.set_pwm(self.BIN2, 0, 4095)
            else:
                pwm.set_pwm(self.BIN1, 0, 4095)
                pwm.set_pwm(self.BIN2, 0, 0)

    def stop_motor(self, motor_id):
        if motor_id == 0:
            pwm.set_pwm(self.PWMA, 0, 0)
        else:
            pwm.set_pwm(self.PWMB, 0, 0)

    def steer(self, angle):
        servo_kit.servo[0].angle = angle

motor = MotorDriver()

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny 임계값을 조정해서 엣지 검출을 더 민감하게 만듦
    canny = cv2.Canny(blur, 30, 100)  # 임계값 조정
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
        if not ret:
            print("카메라에서 영상을 읽을 수 없습니다.")
            break

        canny_image = process_image(frame)
        cropped_image = region_of_interest(canny_image)
        
        # HoughLinesP의 파라미터를 조정하여 더 잘 검출되도록 변경
        lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 30, np.array([]), minLineLength=20, maxLineGap=50)
        line_image = display_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        # 중간 처리 과정 디버깅 (Canny 이미지 및 차선 영역 출력)
        cv2.imshow("Canny Image", canny_image)
        cv2.imshow("Cropped Image", cropped_image)
        cv2.imshow("result", combo_image)

        if lines is not None:
            x1, y1, x2, y2 = lines[0][0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if slope < 0:
                motor.steer(45)  # 좌회전
            else:
                motor.steer(135)  # 우회전
        else:
            motor.steer(90)  # 직진 유지
            motor.stop_motor(0)
            motor.stop_motor(1)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("프로그램 종료: 모터와 서보 모터를 정지 상태로 설정합니다.")
    motor.stop_motor(0)  # 모터 정지
    motor.stop_motor(1)  # 모터 정지
    motor.steer(90)  # 서보 모터를 90도로 조정하여 중립 위치로 설정

finally:
    cap.release()
    cv2.destroyAllWindows()

