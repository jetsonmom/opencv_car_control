# opencv_car_control
<b>   09. 15
<b>  hangul install  https://ahnbk.dev/?p=368
``` bash
 sudo apt install fcitx5 fcitx5-hangul
```
<b> pip install
``` bash
sudo apt-get upgrade
sudo apt-get update
```

``` bash
pip install smbus
pip install adafruit-circuitpython-servokit
pip install Adafruit_PCA9685
pip install Adafruit_MotorHAT
pip install Jetson.GPIO
sudo usermod -aG gpio orin
```

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

오른쪽 차선 비율 보정
오른쪽 차선이 비율이 맞지 않는 문제를 해결하려면, 오른쪽 차선 감지 영역을 더 넓히거나 감지 알고리즘을 조정해줘야 합니다. 이를 위해 몇 가지 방법을 적용할 수 있습니다.

### 1. **ROI (Region of Interest) 영역 확장**
차선 검출 영역(ROI)을 조정해서 오른쪽 영역을 더 넓게 잡는 방법이 있습니다. 현재 ROI는 다음과 같이 설정되어 있습니다:

```python
def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]  # 좌우 영역 설정
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

이 영역을 오른쪽으로 더 넓혀서 오른쪽 차선이 더 잘 감지되도록 할 수 있습니다. 예를 들어:

```python
def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(100, height), (1200, height), (650, 250)]  # 오른쪽을 넓히고 왼쪽을 좁힘
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

위와 같이 `polygons` 배열에서 오른쪽 끝 점을 오른쪽으로 더 넓게 조정하면, 오른쪽 차선이 더 넓은 영역에서 감지될 수 있습니다.

### 2. **Hough Line 검출 파라미터 조정**
차선 검출에서 `cv2.HoughLinesP` 함수의 파라미터를 조정하여 오른쪽 차선을 더 정확히 검출할 수 있습니다. `minLineLength`와 `maxLineGap` 값을 변경해보세요:

```python
lines = cv2.HoughLinesP(cropped_canny, 1, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=10)
```

- `minLineLength` 값을 줄이면 짧은 선도 감지할 수 있습니다.
- `maxLineGap` 값을 높이면 끊긴 부분이 있더라도 더 긴 선으로 이어지도록 할 수 있습니다.

### 3. **오른쪽 차선 비율 보정**
차선이 검출된 후, 오른쪽 차선의 비율이 너무 작다면, 이를 감안해 조향 값을 조정할 수 있습니다. 예를 들어, 오른쪽 차선이 왼쪽보다 적게 감지되면, 조향 각도를 우회전 쪽으로 약간 더 보정해 차가 중심으로 유지되도록 할 수 있습니다.

이를 위해 검출된 차선의 기울기를 계산한 후, 왼쪽 차선과 오른쪽 차선의 기울기 차이를 비교해 서보 모터의 조향을 조정할 수 있습니다:

```python
if lines is not None:
    right_lines = []
    left_lines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
        if slope < 0:
            left_lines.append(line)
        else:
            right_lines.append(line)
    
    if len(right_lines) < len(left_lines):
        Motor.steer(135)  # 오른쪽 비율이 작을 때 우회전 더 크게
    elif len(left_lines) < len(right_lines):
        Motor.steer(45)  # 왼쪽 비율이 작을 때 좌회전 더 크게
    else:
        Motor.steer(90)  # 균형 잡힌 경우 직진
```

### 4. **HSV 기반 노란색 차선 검출**
노란색 차선만 선택적으로 검출하도록 하기 위해 HSV 색상 필터를 사용해 차선을 검출하는 방법도 효과적입니다. 오른쪽 차선이 노란색이라면 다음과 같이 노란색만 필터링할 수 있습니다:

```python
def process_image(img):
    # 이미지를 HSV 색상으로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 노란색 차선 범위 설정 (HSV 색상 범위)
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    
    # 노란색 차선 마스크 생성
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 마스크를 적용하여 노란색 차선만 추출
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # 그레이 스케일 및 엣지 검출
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny
```

이 코드를 적용하면 노란색 차선이 더 잘 검출되어 오른쪽 차선을 인식하는 데 도움이 됩니다.

### 결론
오른쪽 차선의 비율 문제를 해결하기 위해서는:
1. **ROI 영역**을 조정하여 오른쪽 차선 영역을 더 넓히고,
2. **Hough Line 파라미터**를 조정하여 오른쪽 차선을 더 잘 감지하도록 하고,
3. **차선 감지 후** 조향 보정을 적용해 차가 중심을 유지하도록 하거나,
4. **HSV 필터**를 사용해 노란색 차선을 더 명확하게 검출하는 방법을 적용할 수 있습니다.

이러한 수정 사항을 적용하면 오른쪽 차선의 비율 문제를 해결할 수 있을 것입니다.
카메라에서 노란색 차선을 감지할 때, 오른쪽과 왼쪽 차선의 비율이 맞지 않는 경우는 주로 차선 감지 알고리즘에서 차선 검출 영역이나 차선 인식의 정확도가 영향을 미칠 수 있습니다. 이를 해결하기 위해 다음과 같은 부분을 조정할 수 있습니다:

### 1. **ROI (Region of Interest) 조정**
현재 차선을 검출하는 영역(ROI)이 너무 좁거나 편향되어 있을 수 있습니다. `region_of_interest()` 함수에서 차선 감지 영역을 정의하는 다각형을 수정하여 오른쪽과 왼쪽 차선을 더 균등하게 감지하도록 조정할 수 있습니다.

기존의 `region_of_interest()` 함수는 이렇게 정의되어 있습니다:

```python
def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

이 다각형 좌표를 수정하여 좌우 비율을 더 맞추도록 할 수 있습니다. 예를 들어, 왼쪽 차선이 더 감지되지 않는다면 왼쪽 영역을 조금 넓게 잡을 수 있습니다:

```python
def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(100, height), (1200, height), (600, 250)]  # 왼쪽을 더 넓히고, 오른쪽을 약간 좁힘
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

### 2. **차선 검출 알고리즘의 파라미터 조정**
차선 검출에서 `cv2.HoughLinesP` 함수의 파라미터를 조정하는 것도 도움이 될 수 있습니다. 이 함수의 파라미터 중 `minLineLength`, `maxLineGap`이 차선의 감지 정확도에 영향을 미칩니다.

현재 코드는 다음과 같이 되어 있습니다:

```python
lines = cv2.HoughLinesP(cropped_canny, 1, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
```

- `minLineLength`: 검출되는 선의 최소 길이입니다. 값이 작을수록 더 작은 선들도 감지할 수 있습니다.
- `maxLineGap`: 선의 중간에 끊긴 부분이 있어도 이를 이어서 하나의 선으로 감지할 수 있는 최대 간격입니다. 값이 클수록 끊긴 선들을 더 잘 이어줍니다.

예를 들어, 더 긴 차선만 감지하려면 `minLineLength` 값을 높이고, 중간에 끊긴 선을 감지하려면 `maxLineGap` 값을 더 크게 설정할 수 있습니다:

```python
lines = cv2.HoughLinesP(cropped_canny, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=10)
```

### 3. **차선 색상 필터링**
노란색 차선을 감지할 때, 색상 기반으로 필터링을 추가하여 더 정확한 차선 인식을 할 수 있습니다. 노란색 차선을 감지하려면 HSV 색상 공간에서 노란색을 감지하는 필터를 적용하는 방법을 고려해보세요.

```python
def process_image(img):
    # 이미지를 HSV 색상으로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 노란색 차선 범위 설정 (HSV 색상 범위)
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    
    # 노란색 차선 마스크 생성
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 마스크를 적용하여 노란색 차선만 추출
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # 그레이 스케일 및 엣지 검출
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny
```

이 필터링을 추가하면, 노란색 차선만 감지하여 양쪽 차선을 더 정확하게 인식할 수 있습니다.

### 4. **차선 중심 계산**
감지된 차선의 중심을 기준으로 자동차가 얼마나 기울었는지를 계산하여 이를 보정하는 로직을 추가할 수도 있습니다. 예를 들어, 양쪽 차선의 기울기를 계산하고 이를 비교하여, 차가 더 중앙으로 유지되도록 서보를 제어할 수 있습니다.

### 5. **서보 모터 조향 보정**
차선이 한쪽으로 치우쳐져 있으면, 서보 모터의 조향 각도를 조정하여 자동차가 더 균형을 유지하도록 할 수 있습니다. 왼쪽이나 오른쪽 차선이 더 많이 감지되는지에 따라 `steer` 함수를 수정해 차선을 따라가는 로직을 더 정밀하게 할 수 있습니다.

### 결론
1. ROI 영역을 조정하여 양쪽 차선을 더 균등하게 검출.
2. `cv2.HoughLinesP` 함수의 파라미터를 조정하여 차선 검출 성능 향상.
3. HSV 필터를 사용하여 노란색 차선을 더 정확히 검출.
4. 양쪽 차선의 기울기와 중심을 계산하여 차량이 중앙에 있도록 유지.
5. 서보 모터 조향을 통해 차량 방향을 더 정밀하게 제어.

이러한 방법을 통해 차선 인식의 비율 문제를 해결할 수 있습니다.
