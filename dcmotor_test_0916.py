import time
from Adafruit_PCA9685 import PCA9685
from adafruit_servokit import ServoKit

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

try:
    # 직진 설정
    motor.steer(90)  # 서보 모터를 90도로 조정하여 직진
    motor.set_speed(0, 'forward', 50)  # 왼쪽 모터 속도 50
    motor.set_speed(1, 'forward', 50)  # 오른쪽 모터 속도 50

    # 5초 동안 직진
    time.sleep(5)

    # 모터 정지
    motor.stop_motor(0)
    motor.stop_motor(1)

except KeyboardInterrupt:
    print("프로그램 종료: 모터와 서보 모터를 정지 상태로 설정합니다.")
    motor.stop_motor(0)  # 모터 정지
    motor.stop_motor(1)  # 모터 정지
    motor.steer(90)  # 서보 모터를 90도로 조정하여 중립 위치로 설정

