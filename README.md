<div align='center'>
  <img width="388" alt="대회 개요" src="https://github.com/user-attachments/assets/a0718353-0936-4712-a753-43f14274b7fc">
  <h2>🏆 다국어 영수증 OCR</h2>
</div>


<div align="center">

[👀Model](#final-model) |
[🤔Issues](https://github.com/boostcampaitech7/level2-objectdetection-cv-07/issues) | 
[🚀CORD](https://github.com/open-mmlab/mmdetection) |
[🤗Synthetic Data](https://huggingface.co/docs/transformers/en/index)
</div>

## Introduction
주로 AI 모델의 구조나 알고리즘에 집중하기 쉽지만, 실무에서는 데이터의 품질이 모델 성능만큼 중요합니다. 본 대회에서는 Data-Centric AI 접근 방식을 통해, 다국어(중국어, 일본어, 태국어, 베트남어) 영수증 이미지에서 글자를 검출하는 OCR 과제를 수행하고자 합니다.

**Goal :** 쓰레기 객체를 탐지하는 모델을 개발하여 정확한 분리수거와 환경 보호를 지원 <br>
**Data :** UFO 포맷의 글자가 포함된 JPG 이미지 (Train Data 총 400장, Test Data 총 120장)<br>
**Metric :** DetEval(Final Precision, Final Recall, Final F1-Score)

## Project Overview
초기 단계에서는 EDA와 베이스라인 코드 분석을 통해 데이터와 모델에 대한 기초적인 분석을 진행한 후, 외부 및 합성 데이터를 활용하고 데이터 클렌징과 증강 기법을 적용한 다양한 실험을 통해 모델의 일반화 성능을 최적화하였습니다. 최종적으로는 5-fold 앙상블을 적용하여 최적의 성능을 도출하였습니다.<br>
결과적으로 precision:0.9427,	recall:0.8801, f1:0.9103를 달성하여 리더보드에서 4위를 기록하였습니다.<br>

<img width="962" alt="최종 public 리더보드 순위" src="https://github.com/user-attachments/assets/c67163df-3b34-4c5c-aa98-d93612b37d75">

## Model
베이스라인 모델은 EAST (An Efficient and Accurate Scene Text Detector; Zhou et al., 2017)이고, Backbone로는 ImageNet에 사전훈련된 VGG-16 (Visual Geometry Group - 16 layers; Simonyan and Zisserman, 2015)을 사용합니다.




## Data
```
dataset
  ├── annotations
      ├── train.json # train image에 대한 annotation file (coco format)
      └── test.json # test image에 대한 annotation file (coco format)
  ├── train # 4883장의 train image
  └── test # 4871장의 test image
```

## External Data

## File Tree
```
├── .github
├── external_data
    ├── cord
    ├── synthetic_data
├── code
    ├── 
└── README.md
```

## Environment Setting
<table>
  <tr>
    <th colspan="2">System Information</th> <!-- 행 병합 -->
    <th colspan="2">Tools and Libraries</th> <!-- 열 병합 -->
  </tr>
  <tr>
    <th>Category</th>
    <th>Details</th>
    <th>Category</th>
    <th>Details</th>
  </tr>
  <tr>
    <td>Operating System</td>
    <td>Linux 5.4.0</td>
    <td>Git</td>
    <td>2.25.1</td>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10.13</td>
    <td>Conda</td>
    <td>23.9.0</td>
  </tr>
  <tr>
    <td>GPU</td>
    <td>Tesla V100-SXM2-32GB</td>
    <td>Tmux</td>
    <td>3.0a</td>
  </tr>
  <tr>
    <td>CUDA</td>
    <td>12.2</td>
    <td></td>
    <td></td>
  </tr>
</table>
<br>

<p align='center'>© 2024 LuckyVicky Team.</p>
<p align='center'>Supported by Naver BoostCamp AI Tech.</p>

---

<div align='center'>
  <h3>👥 Team Members of LuckyVicky</h3>
  <table width="80%">
    <tr>
      <td align="center" valign="top" width="15%"><a href="https://github.com/jinlee24"><img src="https://avatars.githubusercontent.com/u/137850412?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/stop0729"><img src="https://avatars.githubusercontent.com/u/78136790?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/yjs616"><img src="https://avatars.githubusercontent.com/u/107312651?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/sng-tory"><img src="https://avatars.githubusercontent.com/u/176906855?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/Soojeoong"><img src="https://avatars.githubusercontent.com/u/100748928?v=4"></a></td>
      <td align="center" valign="top" width="15%"><a href="https://github.com/cyndii20"><img src="https://avatars.githubusercontent.com/u/90389093?v=4"></a></td>
    </tr>
    <tr>
      <td align="center">🍀이동진</td>
      <td align="center">🍀정지환</td>
      <td align="center">🍀유정선</td>
      <td align="center">🍀신승철</td>
      <td align="center">🍀김소정</td>
      <td align="center">🍀서정연</td>
    </tr>
    <tr>
      <td align="center">데이터 전처리,Augmentation</td>
      <td align="center">서버 관리, Failure Analysis, 앙상블</td>
      <td align="center">EDA, 데이터 전처리, Augmentation</td>
      <td align="center">데이터 전처리, Augmentation</td>
      <td align="center">스케줄링, 문서화, 데이터 합성</td>
      <td align="center">외부 데이터셋 학습, 깃 관리</td>
    </tr>
  </table>
</div>
