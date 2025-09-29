# 📘 SW중심대학 공동 AI 경진대회 2023 — 위성 이미지 건물 영역 분할

## 1. 대회 링크

[SW중심대학 공동 AI 경진대회 2023 — DACON](https://dacon.io/competitions/official/236092/overview/description)

---

## 2. 대회 개요

* **주제**: 위성 이미지 건물 영역 분할 (Satellite Image Building Area Segmentation)
* **목표**: 위성 이미지를 입력으로 받아, 각 픽셀이 ‘건물 영역’인지 아닌지를 예측하는 세그멘테이션 모델 개발
* **참가 대상**:

  * SW중심대학 소속 학생 (전공 무관, 재학생/휴학생 가능)
  * 졸업생은 참가 불가
* **단체 참가**: 팀 단위 등록 (팀 병합 절차 없음)

---

## 3. 일정 & 주요 마일스톤

| 항목              | 날짜 / 기간 |
| --------------- | ------- |
| 참가 접수 시작        | 05.15   |
| 참가 접수 마감        | 06.23   |
| 팀 병합 마감         | 06.23   |
| 대회 시작           | 07.03   |
| 대회 종료           | 07.28   |
| 코드 및 발표자료 제출 마감 | 08.03   |
| 코드 평가           | 08.10   |
| 전문가 심사 / 온라인 발표 | 08.11   |
| 최종 결과 발표        | 08.14   |
| 시상식             | 08.17   |

---

## 4. 규칙 및 평가 지표

* **평가 방식**: Dice Score 기반 평가 (세그멘테이션 일치도 기준으로 채점)
* **제출 형식**: 예측한 마스크 파일 혹은 모델 출력 형태
* **제한 사항 / 유의 사항**:

  * 데이터 중복 사용 금지
  * 외부 데이터 사용 여부 규정
  * 제출 횟수 제한 존재

---

## 5. 내 프로젝트 개요

* **팀명**: Chat한소리

* **구성원**:
| 이름 | 전공 / 학년 | 역할 | 한 줄 소개 |
| ---- | ---------- | ---- | -------- |
| 문현우 | 컴퓨터공학과 3학년 | 팀장 / 데이터 분석 및 전후처리, 적용 모델 탐색 | 데이터를 분석/전후처리/시각화를 전반적으로 담당하였으며 모델 탐색도 담당하였습니다. |
| 김진우 | 컴퓨터공학과 3학년 | 적용 모델 탐색 및 개량 | 여러 모델을 탐색하고 본 프로젝트에 활용할 수 있도록 개량을 도맡아 하였습니다. |
| 유수민 | 컴퓨터공학과 3학년 | 데이터 분석 및 전처리 | 데이터 처리 관련 업무 중 특히 데이터 분석/전처리를 담당하였습니다. |
| 이유진 | 컴퓨터공학과 3학년 | 데이터 분석 및 전처리 | 프로젝트 전반에서 업무과 과중된 부분을 보조하였습니다. |

* **사용 언어 & 프레임워크**: Python, Jupyter Notebook

* **사용 기술**:

  * PyTorch (DeepUNet 기반 모델 구현)
  * Albumentations (데이터 증강)
  * OpenCV (이미지 전처리 및 후처리)
  * NumPy, Pandas (데이터 처리 및 제출 파일 생성)
  * Matplotlib, tqdm (시각화 및 학습 진행 모니터링)

* **주요 아이디어 / 전략 요약**:

  1. 데이터 전처리 (노이즈 제거, 샤프닝, 밝기/대비 조정)
  2. 데이터 증강 (회전, 크롭, 플립, Normalize, Resize)
  3. 모델: DeepUNet / DeepLabV3+ 기반 실험
  4. 손실 함수: BCEWithLogitsLoss + 클래스 가중치 적용
  5. 후처리 기법: Thresholding, Morphological 연산 (Open/Close)

* **성능 결과**:

  * 최종 점수: 0.71134 / 1

* **리더보드 순위 및 소감**: 중위권 기록 (97/227). 데이터 증강과 후처리가 성능 향상에 크게 기여하였으나 인공지능 배경지식이 부족한 상태로 대회에 참여하였기 때문에 성능 향상에 매우 중요한 역할을 하는 앙상블 기법 등을 활용하지 못하였던 것이 아쉬웠음.

---

## 6. 코드 구조

```
/  
├── data/  
│   ├── train_images/  
│   ├── train_masks/  
│   ├── test_images/  
│   └── ...  
├── notebooks/  
│   └── exploratory_analysis.ipynb  
├── src/  
│   ├── dataset.py  
│   ├── model.py  
│   ├── train.py  
│   ├── predict.py  
│   └── utils.py  
├── outputs/  
│   ├── submissions/  
│   └── logs/  
├── requirements.txt  
├── README.md  
└── LICENSE  
```

---

## 7. 실행 방법

1. 필요한 라이브러리 설치

   ```bash
   pip install -r requirements.txt
   ```

2. 데이터 전처리 실행

   ```bash
   python src/dataset.py --mode preprocess
   ```

3. 모델 학습

   ```bash
   python src/train.py --config configs/train_config.yaml
   ```

4. 테스트 예측

   ```bash
   python src/predict.py --model_path outputs/best_model.pth --output submissions/
   ```

5. 제출 파일 생성

   ```bash
   # 예: submissions/prediction.csv 또는 이미지 마스크 형태 제출
   ```

---

## 9. 참고 리소스

* [DACON 대회 페이지](https://dacon.io/competitions/official/236092/overview/description)
* Segmentation 관련 논문 및 블로그 자료
* GitHib 오픈소스 코드 (U-Net, DeepLab 등)
* Albumentations 관련 문서

---

## 10. 향후 개선 가능성

* 앙상블 기법을 활용하여 복수의 모델을 활용한다면 큰 폭의 성능 향상을 기대할 수 있을 것으로 보임

---
