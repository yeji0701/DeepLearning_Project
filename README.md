Deep Learning Project

Tracking people in CCTV using YOLO
==================================

I. 프로젝트 개요
----------------
1. 프로젝트 배경 및 목적
현대사회에서 CCTV는 보안 및 치안 등의 목적으로 곳곳에 설치되어 있다. 사람들의 편리와 안전을 지켜주기 위해 하루에도 엄청난 양의 데이터를 쏟아내고 있지만, 미아가 생기고 범죄 사건이 발생한다. 본 프로젝트에서는, YOLO를 활용한 Object Tracking으로 미아 혹은 수상한 사람을 CCTV에서 검출하여 다른 각도의 CCTV에서 또한 발견한 인물을 검출하도록 노력하였다. YOLO의 여러 버전들을 비교 분석하며, 목표 인물이 영상에 검출 되었다면 실제 사람이 해당 영상을 통해 확인하고 추적하는 컨셉이기 때문에 끊기지 않는 추적보다는 목표 인물 검출에 포커스를 두었다.
   
2. 데이터셋 소개
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108450701-fa0a8e00-72a8-11eb-82f0-0d82da5d6924.png" width="780" height="280"></p>

- 실내 50개, 실외 50 총 100개의 폴더로 구성
* 폴더 구성
  * video : 추적대상이 찍힌 영상 파일(각 폴더마다 3~5개의 
  * json : json형식의 bounding box 좌표
  * frames : 추적대상의 이미지 파일
   
 
II. 프로젝트 결과
-----------------
- 실내 17개의 frame 학습 후 test
- 실외 18개의 frame 학습 후 test
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108315152-5a90c100-71fe-11eb-82eb-712fbe3c8ca2.gif" width="390" height="230"/> <img src="https://user-images.githubusercontent.com/72811950/108314491-5617d880-71fd-11eb-925d-a49820d311f0.gif" width="390" height="230"/></p>

- 프로젝트 전체 결과 영상  
  indoor : <https://youtu.be/EPoV2Pz7U2Y>  
  outdoor : <https://youtu.be/Uwu12zHNlns>

III. Process
-------------
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108320197-d17d8800-7205-11eb-9265-297ef37e5a0a.png" width="780" height="180"></p>

1. Preprocessing
   * Image Augmentation
      <p align="center"><img src="https://user-images.githubusercontent.com/72811950/108452462-e3196b00-72ab-11eb-9472-0caae061ef4a.jpg" width="780" height="400"></p>
      - [Image augmentation code](기중 이미지 증강 시키는 코드 커밋하고 여기에 코드 url 넣어주세여)

   * json -> txt
      ![image](https://user-images.githubusercontent.com/28764376/108456228-37741900-72b3-11eb-87ad-d6dab055b416.png)
      - [Format conversion code](https://github.com/yeji0701/DeepLearning_Project/blob/main/code/jc/00_label_json_to_txt.ipynb)

2. Training
   * Changing Resolution Size
   ```
   <yolo.cfg>

   # 정확도 향상을 위해 픽셀 해상도를 크게 함

   batch=64
   subdivisions=32
   width=608  <-- 변경
   height=608  <-- 변경
   ```
   * Optimizing Anchor Box
   ```
   여기에도 뭐가 들어갈거죵?
   ```
3. Test
   * Adjusting Confidence Threshold
   ```
   # -threshold {} <-- 조정하여 되도록 target만 detection하도록 함
   
   ./darknet detector demo custom_data/detector.data custom_data/cfg/yolov3-custom-test.cfg 
   backup/yolov3-custom_best.weights ./test.mp4 - thresh 0.6 -out_filename out.avi -dont_show
   ```

마치며
------
- 배운점
1. 다양한 이미지 증강 기법  
2. Object detection 알고리즘의 발전  
3. YOLO 모델의 원리 학습  
4. IoU, mAP 등 Object detection 평가 지표에 대한 이해

- 개선할 점
1. yolo 소스 코드 분석을 통한 데이터에 최적화 된 튜닝
2. 보다 Target에 특화된 Custom Training을 통한 모델 개선
3. 모든 시도는 소중하니, 결과를 기록하는 습관 개선
