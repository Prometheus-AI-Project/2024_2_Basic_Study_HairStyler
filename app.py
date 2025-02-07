import streamlit as st
from RAG_hairCLIP import get_hairStyleColor  # LLM 관련 함수 불러오기

# ✅ Streamlit UI 구성
st.title("💇‍♀️ 헤어스타일 추천 AI")

st.write("당신에게 어울리는 헤어스타일과 컬러를 추천해드립니다. 원하는 스타일을 입력해보세요!")

# 텍스트 입력 필드
user_input = st.text_input("헤어스타일 관련 질문을 입력하세요:", placeholder="예: 긴 머리에 어울리는 색상은?")
submit_button = st.button("추천 받기")

# 버튼 클릭 시 결과 표시
if submit_button and user_input:
    with st.spinner("AI가 스타일을 추천하는 중...💡"):
        result = get_hairStyleColor(user_input)
        st.subheader("🎨 추천 결과")
        st.write(result)
