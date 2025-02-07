## 🔥2024_2_Basic_Study_HairStyler
2024년 2학기 기초스터디 3팀 토이 프로젝트

## 🤔프로젝트 소개
🥺 평소에 헤어 스타일에 대한 고민이 많으신가요?<br>
😮 자신의 이목구비와 얼굴 형과 같은 얼굴의 특징을 입력만 해준다면!<br>
😎 최적의 스타일을 추천해줌과 동시에 사용자의 사진을 해당 헤어 스타일로 바꿔주어 얼마나 잘 어울리는 지 미리 확인이 가능합니다!<br>

## 💻파이프라인
* 얼굴형, 이목구비에 따라 어울리는 헤어 스타일을 정리한 웹사이트에서 텍스트 크롤링
* 임베딩을 위한 텍스트 분할 이후 벡터 데이터베이스 구축
* 사용자가 헤어스타일에 대한 고민을 Question Query로 GPT-4o에 입력 -> multi-query 방식으로 다양한 Question Query Variant들을 생성하여 답변의 융통성을 부여
* 프롬프트 엔지니어링을 통해 GPT-4o가 헤어 스타일과 헤어 컬러로 나눠서 답변을 수행하도록 제한
* 두 가지 텍스트 키워드(헤어스타일, 헤어 컬러)를 HairCLIPv2 모델에 넣어 사용자 얼굴에 추천된 헤어스타일을 생성

## ⭐Demo Day
* 2025/02/08 프로메테우스 데모 데이 부스 운영
* streamlit을 활용해 간단한 웹페이지를 구축한 후 헤어스타일 추천 및 사진 생성 서비스 진행
<img width="774" alt="Image" src="https://github.com/user-attachments/assets/1cbf3ea5-22d5-459f-8dca-ccce7cb913e5" />


## 😎Members
| 심수민 (팀장, 개발)      | 강민진 (개발)     | 김성재 (개발)     | 조현진 (개발)  |
|:-----------------:|:----------------:|:-----------------:|:--------------------:|
| 2기      | 6기 | 6기 | 6기 |
| [use08174](https://github.com/use08174)        |  [Minjin03](https://github.com/Minjin03)  |  [jayimnida](https://github.com/jayimnida)   |  [hyun-jin891](https://github.com/hyun-jin891)|



