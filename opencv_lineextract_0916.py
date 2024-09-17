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
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blur, 30, 100)  # Canny 엣지 검출
    return canny

def region_of_interest(img):
    height, width = img.shape[:2]  # 이미지의 높이와 너비를 가져옴
    polygons = np.array([
    [(0, height), (width, height), (width // 2, height // 2 - 160)]
    ])

    mask = np.zeros_like(img)  # 이미지와 같은 크기의 검은색 마스크 생성
    cv2.fillPoly(mask, polygons, 255)  # 다각형을 흰색으로 채움
    
    # 다각형을 원본 이미지에 그려서 시각적으로 확인
    cv2.polylines(img, polygons, isClosed=True, color=(0, 255, 0), thickness=3)

    masked_img = cv2.bitwise_and(img, mask)  # 마스크와 이미지를 AND 연산하여 관심 영역만 남김
    cv2.imshow("76 Mask", mask)  # 마스크 자체를 출력하여 확인
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
        lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 70, np.array([]), minLineLength=20, maxLineGap=10)
        line_image = display_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        # 중간 처리 과정 디버깅 (Canny 이미지 및 차선 영역 출력)
        cv2.imshow("108 canny Image", canny_image)
        cv2.imshow("109 Cropped Image", cropped_image)
        cv2.imshow("result", combo_image)

        if lines is not None:
            x1, y1, x2, y2 = lines[0][0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if slope < -0.5:  # 좌회전
                motor.steer(45)  # 좌회전
                motor.set_speed(1, 'forward', 100)  # 좌회전 시 모터 속도 설정
                motor.set_speed(0, 'forward', 100)
            elif slope > 0.5:  # 우회전
                motor.steer(135)  # 우회전
                motor.set_speed(1,'forward', 100)  # 우회전 시 모터 속도 설정
                motor.set_speed(0,'forward', 100)
            else:  # slope가 거의 0일 때 직진
                motor.steer(90)  # 직진을 위한 각도 유지
                motor.set_speed(1, 'forward', 100)  # 직진 속도 설정
                motor.set_speed(0, 'forward', 100)  # 양쪽 모터를 동일 속도로 설정
               
        else:
                motor.set_speed(1, 'forward', 0)  # 직진 속도 설정
                motor.set_speed(0, 'forward', 0)  # 양쪽 모터를 동일 속도로 설정

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

