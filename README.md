<h3 align="center"><strong>Deep Learning Project</strong></h1>
<h1 align="center"><strong>Object Detection Project</strong></h1>

## 프로젝트 목적
- 객체를 추적하는 신경망(YOLO)의 원리와 방식을 이해하고 사용법 학습

## 프로젝트 목표
- 본 프로젝트는 CCTV에서 특정 타겟을 검출하는데 목표가 있다.  
  모든 object를 추적하는 것이 아니라  특정 인물을 학습하고 다른 CCTV에서 그 인물이 나타났을 때 검출을 하도록 목표를 세웠다.  
  이 경우 추적 범위를 빠르게 좁혀 미아찾기나 보안 등에 활용될 수 있다.
 
## 프로젝트 결과
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108315152-5a90c100-71fe-11eb-82eb-712fbe3c8ca2.gif" width="390" height="230"/> <img src="https://user-images.githubusercontent.com/72811950/108314491-5617d880-71fd-11eb-925d-a49820d311f0.gif" width="390" height="230"/></p>

- 프로젝트 전체 결과 영상  
  indoor : <https://youtu.be/EPoV2Pz7U2Y>  
  outdoor : <https://youtu.be/Uwu12zHNlns>

## Dataset
- Dataset은 실내와 실외로 구분되어 있고 실내 50개 실외 50개로 총 100개의 폴더가 있다.  
  실내는 market 내부 영상이며 실외는 거리를 찍은 영상으로 폴더마다 다른 곳에서 찍힌 target의 video와 json형식의 라벨이 포함되어 있다.
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108324817-bca3f300-720b-11eb-8232-5a91d173b46d.png" width="550" height="280"></p>


## Process
<p align="center"><img src="https://user-images.githubusercontent.com/72811950/108320197-d17d8800-7205-11eb-9265-297ef37e5a0a.png" width="780" height="180"></p>

> ### Preprocessing
> - Image Augmentation
> ```
> # 이미지파일 읽기
> def read_train_dataset(dir):
>   images = []
>   names = []
> 
>   for file in listdir(dir):
>     if 'jpg' in file.lower() or 'png' in file.lower():
>       images.append(cv2.imread(dir + file, 1))
>       name = file.split('.')[0]
>       names.append(name)
>     images = np.array(images)
>     return images, names
> 
> # 다양한 이미지 증강 효과 설정 
> seq_list = [iaa.Flipud(1.0), iaa.Fliplr(1.0), iaa.AverageBlur(k=5), 
>             iaa.imgcorruptlike.GaussianNoise(severity=2),
>             iaa.imgcorruptlike.Brightness(severity=2), iaa.SaltAndPepper(0.1), iaa.Grayscale(alpha=1.0), 
>             iaa.Affine(rotate=45), iaa.Affine(rotate=-30), iaa.Affine(shear=(-25, 25)), 
>             iaa.PiecewiseAffine(scale=(0.01, 0.05)), iaa.Rot90(1), iaa.imgcorruptlike.Snow(severity=2)]
> 
> ia.seed(1)
> dir = 'dir'
> images, names = read_train_dataset(dir)
> 
> # 증강한 이미지 저장
> for count, seq in enumerate(seq_list):
> 
>   for idx in range(len(images)):
>     image = images[idx]
>     seq = seq
>     seq_det = seq.to_deterministic()
>     image_aug = seq_det.augment_images([image])[0]
>     new_image_file = "dir/after_{}_{}.jpg".format(count, names[idx])
>     cv2.imwrite(new_image_file, image_aug)
> ```
> ### Training
> - Changing Resolution Size
> ```
> # 정확도 향상을 위해 픽셀 해상도를 크게 함
> 
> batch=64
> subdivisions=32
> width=608  <-- 변경
> height=608  <-- 변경
> ```
> - Optimizing Anchor Box
> ```
>
> ```
> ### Test
> - Adjusting Confidence Threshold
> ```
> # -threshold {} <-- 조정하여 되도록 target만 detection하도록 함
> 
> ./darknet detector demo custom_data/detector.data custom_data/cfg/yolov3-custom-test.cfg 
> backup/yolov3-custom_best.weights ./test.mp4 - thresh 0.6 -out_filename out.avi -dont_show
> ```

## 마치며
- 배운점
1. 다양한 이미지 증강 기법  
2. Object detection 알고리즘의 발전  
3. YOLO 모델의 원리 학습  
4. IoU, mAP 등 Object detection 평가 지표에 대한 이해

- 개선할 점
1. yolo 소스 코드 분석을 통한 데이터에 최적화 된 튜닝
2. 보다 Target에 특화된 Custom Training을 통한 모델 개선
3. 모든 시도는 소중하니, 결과를 기록하는 습관 개선
