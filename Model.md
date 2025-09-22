
## **MoveNet 자세 분석 AI 모델의 작동 원리**

### 1. **전체적인 처리 과정**

```
웹캠 영상 → 전처리 → AI 모델 → 후처리 → 키포인트 시각화
```

### 2. **상세한 작동 단계**

#### **단계 1: 입력 데이터 전처리**
```javascript
// 1. 웹캠 영상을 캡처
const video = document.getElementById('video');

// 2. 영상을 Tensor로 변환
const input = tf.browser.fromPixels(video)
  .resizeBilinear([192, 192])  // 192x192 크기로 리사이즈
  .expandDims(0)               // 배치 차원 추가 [1, 192, 192, 3]
  .div(255.0)                  // 0-1 범위로 정규화
  .mul(255)                    // 0-255 범위로 변환
  .toInt();                    // int32 타입으로 변환
```

#### **단계 2: AI 모델 구조 (MoveNet Lightning)**

**입력**: `[1, 192, 192, 3]` (배치, 높이, 너비, RGB 채널)
**출력**: `[1, 6, 56]` (배치, 사람 수, 키포인트 정보)

**모델 아키텍처**:
- **Backbone**: MobileNet 기반 경량화 네트워크
- **Feature Extraction**: CNN 레이어들로 특징 추출
- **Pose Estimation**: 17개 키포인트 예측

#### **단계 3: 키포인트 정보 구조**

각 키포인트는 3개 값으로 구성:
```javascript
// 17개 키포인트 × 3개 값 = 51개 값
// + 5개 추가 정보 = 총 56개 값

for (let i = 0; i < 17; i++) {
  const x = keypoints[i * 3];        // X 좌표 (0-1 범위)
  const y = keypoints[i * 3 + 1];    // Y 좌표 (0-1 범위)  
  const confidence = keypoints[i * 3 + 2]; // 신뢰도 (0-1 범위)
}
```

### 3. **AI 모델의 핵심 기술**

#### **A. Convolutional Neural Network (CNN)**
```
입력 이미지 → Conv2D → BatchNorm → ReLU → MaxPool
           ↓
특징 맵 → 더 깊은 레이어들 → 최종 특징 벡터
```

#### **B. Pose Estimation Head**
```
특징 벡터 → Dense Layer → 17개 키포인트 예측
```

#### **C. Multi-Person Detection**
- 최대 6명까지 동시 감지 가능
- 각 사람별로 17개 키포인트 예측

### 4. **실시간 처리 파이프라인**

```javascript
async function predict() {
  // 1. 비디오 프레임 캡처
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  // 2. 전처리
  const input = tf.browser.fromPixels(video)
    .resizeBilinear([192, 192])
    .expandDims(0)
    .div(255.0)
    .mul(255)
    .toInt();
  
  // 3. AI 모델 추론
  const predictions = await model.predict(input);
  const keypoints = await predictions.data();
  
  // 4. 후처리 및 시각화
  drawKeypoints(ctx, keypoints, 192, 192, canvas.width, canvas.height);
  
  // 5. 메모리 정리
  input.dispose();
  predictions.dispose();
  
  // 6. 다음 프레임 처리
  requestAnimationFrame(predict);
}
```

### 5. **키포인트 시각화 원리**

```javascript
function drawKeypoints(ctx, keypoints, modelWidth, modelHeight, canvasWidth, canvasHeight) {
  // 좌표 스케일링
  const scaleX = canvasWidth / modelWidth;
  const scaleY = canvasHeight / modelHeight;
  
  // 17개 키포인트 그리기
  for (let i = 0; i < 17; i++) {
    const x = keypoints[i * 3] * scaleX;        // 모델 좌표 → 화면 좌표
    const y = keypoints[i * 3 + 1] * scaleY;
    const confidence = keypoints[i * 3 + 2];
    
    // 신뢰도가 높은 키포인트만 그리기
    if (confidence > 0.3) {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}
```

### 6. **성능 최적화 기술**

#### **A. 모델 경량화**
- **MobileNet Backbone**: 모바일 환경에 최적화
- **Quantization**: 8비트 정수 연산 사용
- **Pruning**: 불필요한 가중치 제거

#### **B. 실시간 처리**
- **TensorFlow.js**: GPU 가속 지원
- **메모리 관리**: Tensor 자동 해제
- **배치 처리**: 단일 프레임 처리

### 7. **17개 키포인트의 의미**

```
0: 코 (Nose)
1-2: 눈 (Left/Right Eye)  
3-4: 귀 (Left/Right Ear)
5-6: 어깨 (Left/Right Shoulder)
7-8: 팔꿈치 (Left/Right Elbow)
9-10: 손목 (Left/Right Wrist)
11-12: 엉덩이 (Left/Right Hip)
13-14: 무릎 (Left/Right Knee)
15-16: 발목 (Left/Right Ankle)
```

### 8. **실제 동작 과정**

1. **웹캠 영상 캡처** → RGB 픽셀 데이터
2. **전처리** → 192×192 크기로 리사이즈, 정규화
3. **AI 추론** → CNN으로 특징 추출 → 키포인트 예측
4. **후처리** → 좌표 변환, 신뢰도 필터링
5. **시각화** → 빨간 점으로 키포인트 표시
6. **반복** → 다음 프레임 처리

### 9. **기술적 특징**

- **실시간 처리**: 30fps 이상 성능
- **다중 인물**: 최대 6명 동시 감지
- **경량화**: 모바일 환경 지원
- **정확도**: 높은 키포인트 감지 정확도

이렇게 MoveNet은 **CNN 기반의 실시간 자세 분석 AI**로, 웹캠 영상을 입력받아 사람의 17개 주요 관절을 실시간으로 감지하고 시각화하는 시스템입니다!