# opencv_car_control
#### 9월 8일
<b> 1. opencv_car시작하기
``` bash
mkdir opencv_car
cd opencv_car
sudo apt install python3.10-venv  # python3-venv 패키지 설치: 
cd ~/opencv_car  # 올바른 디렉터리에 있는지 확인
python3 -m venv opencv_car  # 환경 이름을 올바르게 입력하세요 프로젝트 디렉터리로 돌아가 가상 환경을 다시 생성
source opencv_car/bin/activate  # 가상 환경 활성화:
pip install opencv-python #가상 환경이 활성화된 상태에서 OpenCV를 설치

```

<b> chatgpt 자료
이제 USB 카메라를 사용하여 OpenCV로 차선을 인식하고 자동차를 움직이는 코드를 준비하겠습니다. 여기서는 기존에 제공한 자동차 제어 코드와 함께 차선 인식 로직을 통합하여 자동차가 차선을 따라 운전할 수 있도록 설정합니다.

### 필요한 도구 및 라이브러리
1. **OpenCV**: 이미지 처리 및 차선 인식
2. **NumPy**: 데이터 처리
3. **USB 카메라**: 실시간 영상 취득

### 설치
OpenCV를 설치하려면 다음 명령어를 사용합니다:
```bash
pip install opencv-python
```

### 차선 인식과 자동차 제어 통합 코드

```python
import cv2
import numpy as np
import time
# 자동차 제어 코드에서 사용된 모듈 및 클래스 임포트
from your_motor_control_module import MotorDriver, Motor, servo_kit  # 예시 경로와 모듈 이름

# OpenCV 카메라 설정
cap = cv2.VideoCapture(0)  # 카메라 디바이스 번호, USB 카메라가 0번일 수 있음

def process_image(img):
    # 이미지를 그레이 스케일로 변환하고 엣지 검출
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(img):
    # 차선 검출에 필요한 이미지의 특정 영역을 정의
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def display_lines(img, lines):
    # 검출된 선을 이미지 위에 표시
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

while True:
    ret, frame = cap.read()
    if ret:
        canny_image = process_image(frame)
        cropped_canny = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_canny, 1, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
        line_image = display_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", combo_image)
        
        # 기본적인 차선 유지 로직
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                # 간단한 예시: 선의 경사를 기반으로 좌회전 또는 우회전 결정
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                if slope < 0:
                    Motor.steer(45)  # 좌회전
                else:
                    Motor.steer(135) # 우회전
        else:
            Motor.steer(90)  # 직진
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

### 코드 실행 및 테스트
위 코드를 실제 차량에 적용하기 전에, 시뮬레이션 환경에서 먼저 테스트하는 것이 좋습니다. 이를 통해 차량의 반응을 보고 필요한 조정을 할 수 있습니다. 차선 인식 기반의 자동차 제어는 다양한 환경 변수(날씨, 조명 등)에 따라 성능 차이가 클 수 있으므로, 충분한 테스트 후 실제 환경에 적용하세요.
