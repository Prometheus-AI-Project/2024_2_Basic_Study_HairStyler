## 🔥2024_2_Basic_Study_HairStyler
2024년 2학기 기초스터디 3팀 토이 프로젝트

## 🤔프로젝트 소개
🥺 평소에 헤어 스타일에 대한 고민이 많으신가요?<br>
😮 자신의 이목구비와 얼굴형과 같은 특징을 입력만 해준다면!<br>
😎 최적의 스타일을 추천해줌과 동시에 사용자의 사진을 해당 헤어스타일로 바꿔주어 얼마나 잘 어울리는 지 미리 확인이 가능합니다!<br>

## 💻파이프라인
* 얼굴형, 이목구비에 따라 어울리는 헤어스타일을 정리한 웹사이트에서 텍스트 크롤링
* 임베딩을 위한 텍스트 분할 이후 벡터 데이터베이스 구축
* 사용자가 헤어스타일에 대한 고민을 Question Query로 GPT-4o에 입력 -> multi-query 방식으로 다양한 Question Query Variant들을 생성하여 답변의 융통성을 부여
* 프롬프트 엔지니어링을 통해 GPT-4o가 헤어스타일과 헤어 컬러로 나눠서 답변을 수행하도록 제한
* 두 가지 텍스트 키워드(헤어스타일, 헤어 컬러)를 HairCLIPv2 모델에 넣어 사용자 얼굴에 추천된 헤어스타일을 생성

## 📟RAG Experiment
* Question Query 생성<br>
  ![Image](https://github.com/user-attachments/assets/1c264df0-69f3-401b-8044-210b252c2d18)
* 헤어스타일과 헤어 컬러를 나눠서 답변이 생성되도록 프롬프트 엔지니어링<br>
  ![Image](https://github.com/user-attachments/assets/d005400a-e861-4fa0-8536-86f30225e004)

## 💾HairCLIPv2 Image Generation Experiment
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/19d92a38-34da-4099-b96a-797915948c5b" />
<img width="632" alt="Image" src="https://github.com/user-attachments/assets/56b8154e-1dcf-484f-a47e-ad53e7a9243e" />
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/e639a5e6-b648-4610-ba73-1ceea5c107b9" />
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/7e708e07-9b32-4959-b008-93bfa723cda5" />
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/eef406ad-902b-4d3e-8e65-e53bdc0ef4b1" />
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/f61399ba-031d-4b8f-b86e-27ffb47318c4" />
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/15a07d07-095d-47a5-b670-87d079d5ba16" />
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/067aa82b-1d47-4bac-b8da-9a4360d32b72" />

## ⭐Demo Day
* 2025/02/08 프로메테우스 데모 데이 부스 운영
* streamlit을 활용해 간단한 웹페이지를 구축한 후 헤어스타일 추천 및 사진 생성 서비스 진행
<img width="774" alt="Image" src="https://github.com/user-attachments/assets/1cbf3ea5-22d5-459f-8dca-ccce7cb913e5" />
<br>
* 헤어스타일 질문 query 생성 및 답변 생성 (헤어컬러는 프롬프트에서 뺀 상태)
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/3e483068-163d-4a71-9c58-c8e95cc8f7c3" />
<br>
* 헤어스타일 생성
<br>
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/6514d65a-ea0f-4066-8c6b-3cebbf1fdf40" />
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/ea0a42f4-19fa-45a6-858c-bcdffb18f29e" />

## 😎Members
| 심수민 (팀장, 개발)      | 강민진 (개발)     | 김성재 (개발)     | 조현진 (개발)  |
|:-----------------:|:----------------:|:-----------------:|:--------------------:|
| 2기      | 6기 | 6기 | 6기 |
| [use08174](https://github.com/use08174)        |  [Minjin03](https://github.com/Minjin03)  |  [jayimnida](https://github.com/jayimnida)   |  [hyun-jin891](https://github.com/hyun-jin891)|


