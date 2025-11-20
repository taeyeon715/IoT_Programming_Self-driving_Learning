# 🚗 IoT 자율주행 자동차 학습 프로젝트

라즈베리 파이 기반 딥러닝 자율주행 자동차 시스템

---

## 📌 프로젝트 개요

이 프로젝트는 **라즈베리 파이**와 **딥러닝 모델**을 활용한 자율주행 자동차를 구현한 IoT 프로그래밍 프로젝트입니다. 카메라로 차선을 인식하고 NVIDIA CNN 모델로 조향각을 예측하여 자율주행을 수행하며, 물체 감지를 통해 안전 기능도 구현했습니다.

---

## 🎯 주요 기능

### 1️⃣ **데이터 수집 및 학습**
- 수동 조작으로 주행 데이터 수집
- 이미지와 조향각 데이터 라벨링 자동화
- NVIDIA End-to-End Learning 모델 학습

### 2️⃣ **자율주행 시스템**
- 실시간 차선 추적 및 조향각 예측
- 전처리된 이미지 기반 딥러닝 추론
- PWM 모터 제어를 통한 정밀 주행

### 3️⃣ **물체 감지 및 안전 기능**
- OpenCV DNN + SSD MobileNet V2 COCO 모델
- 80개 이상의 객체 클래스 실시간 인식
- 사람 감지 시 자동 정지 기능

---

## 🛠️ 기술 스택

### **하드웨어**
- 라즈베리 파이 (Raspberry Pi)
- 카메라 모듈
- DC 모터 × 2
- L298N 모터 드라이버

### **소프트웨어**
| 분야 | 기술 |
|------|------|
| **프로그래밍 언어** | Python 3.x |
| **딥러닝 프레임워크** | TensorFlow 1.14.0, Keras 2.2.4 |
| **컴퓨터 비전** | OpenCV, OpenCV DNN |
| **하드웨어 제어** | RPi.GPIO |
| **데이터 처리** | NumPy, Pandas, Scikit-learn |
| **모델** | NVIDIA CNN, SSD MobileNet V2 COCO |

---

## 📂 프로젝트 구조

```
IoT_SelfDriving_Learning_Project/
│
├── 자동차 학습용 소스코드_6조.py      # 데이터 수집 (수동 주행)
├── 자동차 주행용 소스코드_6조.py      # 자율주행 실행
├── 물체탐지 소스코드_6조.py           # 자율주행 + 물체 감지
├── make_model.ipynb                   # 딥러닝 모델 학습 노트북
└── README.md                          # 프로젝트 문서
```

---

## 🚀 사용 방법

### **1단계: 학습 데이터 수집**

```bash
python "자동차 학습용 소스코드_6조.py"
```

**키보드 조작**:
- `방향키 ↑`: 전진
- `방향키 ↓`: 정지
- `방향키 ←`: 좌회전 (각도 45도로 저장)
- `방향키 →`: 우회전 (각도 135도로 저장)
- `q`: 종료

수집된 이미지는 `/home/dragon/AI_CAR/video/train/` 폴더에 자동 저장됩니다.

---

### **2단계: 딥러닝 모델 학습**

```bash
jupyter notebook make_model.ipynb
```

**학습 과정**:
1. `./train_3` 폴더에 수집한 학습 데이터 배치
2. 노트북의 "Run All" 실행
3. `lane_navigation_final.h5` 모델 파일 생성

**모델 구조**:
- 입력: 90×360×3 (전처리된 이미지)
- 출력: 조향각 (45~135도)
- 에폭: 10
- 배치 크기: 100
- 성능: R² = 91.77%

---

### **3단계: 자율주행 실행**

#### **기본 자율주행**

```bash
python "자동차 주행용 소스코드_6조.py"
```

#### **물체 감지 기능 포함**

```bash
python "물체탐지 소스코드_6조.py"
```

**키보드 조작**:
- `방향키 ↑`: 자율주행 시작
- `방향키 ↓`: 정지
- `q`: 프로그램 종료

---

## 🧠 딥러닝 모델 상세

### **NVIDIA End-to-End Learning 아키텍처**

```
입력 이미지 (90×360×3)
    ↓
Conv2D (24 filters, 5×5, stride 2×2) + ELU
    ↓
Conv2D (36 filters, 5×5, stride 2×2) + ELU
    ↓
Conv2D (48 filters, 5×5, stride 2×2) + ELU
    ↓
Conv2D (64 filters, 3×3) + ELU
    ↓
Dropout (0.2)
    ↓
Conv2D (64 filters, 3×3) + ELU
    ↓
Flatten
    ↓
Dropout (0.2)
    ↓
Dense (100) + ELU
    ↓
Dense (50) + ELU
    ↓
Dense (10) + ELU
    ↓
Dense (1) → 조향각 출력
```

### **이미지 전처리 파이프라인**

1. **이미지 하단 절반 추출**: 차선 정보가 있는 하단만 사용
2. **색상 공간 변환**: BGR → YUV
3. **가우시안 블러**: 노이즈 제거 (커널 크기 5×5)
4. **임계값 처리**: 차선 강조 (threshold=160)
5. **크기 조정**: 360×90 픽셀
6. **정규화**: 0~1 범위로 스케일링

---

## 🎬 물체 감지 시스템

### **SSD MobileNet V2 COCO 모델**

- **프레임워크**: OpenCV DNN
- **모델 파일**:
  - `frozen_inference_graph.pb`
  - `ssd_mobilenet_v2_coco_2018_03_29.pbtxt`
- **인식 가능 객체**: 80+ 클래스 (사람, 자동차, 자전거, 신호등 등)
- **신뢰도 임계값**: 50%

### **안전 기능**

- 사람이 감지되면 자동으로 `carState = "stop"` 전환
- 멀티스레드로 차선 추적과 물체 감지 동시 수행

---

## ⚙️ 하드웨어 설정

### **GPIO 핀 배치**

| 모터 | PWM 핀 | 방향 핀 1 | 방향 핀 2 |
|------|--------|----------|----------|
| **왼쪽 모터** | GPIO 18 (PWMA) | GPIO 22 (AIN1) | GPIO 27 (AIN2) |
| **오른쪽 모터** | GPIO 23 (PWMB) | GPIO 25 (BIN1) | GPIO 24 (BIN2) |

### **모터 제어 함수**

```python
motor_go(speed)     # 전진
motor_back(speed)   # 후진
motor_left(speed)   # 좌회전
motor_right(speed)  # 우회전
motor_stop()        # 정지
```

---

## 📊 학습 결과

- **학습 데이터**: 2,932개
- **검증 데이터**: 733개
- **최종 손실값**: 46.86 (Training), 83.29 (Validation)
- **R² Score**: 91.77% ✅
- **MSE**: 66

---

## 🎥 시연 영상

*(시연 영상이 있다면 여기에 추가)*

---

## 📖 참고 자료

- [NVIDIA End-to-End Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)
- [OpenCV DNN Documentation](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

---

## 👥 팀 정보

**6조** - IoT 시스템 설계 및 실습 프로젝트

---

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

## 🔧 문제 해결

### **카메라가 인식되지 않을 때**
```bash
vcgencmd get_camera
# supported=1 detected=1 확인
```

### **모델 파일이 없을 때**
```bash
# make_model.ipynb를 실행하여 lane_navigation_final.h5 생성
```

### **GPIO 권한 오류**
```bash
sudo python "자동차 주행용 소스코드_6조.py"
```

---

## 🌟 향후 개선 방향

- [ ] 실시간 성능 최적화
- [ ] 더 많은 객체 클래스 감지
- [ ] 신호등 인식 기능 추가
- [ ] 장애물 회피 알고리즘 구현
- [ ] 웹 기반 원격 제어 인터페이스

---

**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**


