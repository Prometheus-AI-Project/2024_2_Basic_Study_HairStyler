import streamlit as st
from Hairclipv2 import hairstyle_editing_pipeline
from PIL import Image

# ✅ Streamlit UI 설정
st.title("💇‍♀️ 헤어스타일 변경 AI")
st.write("원하는 헤어스타일을 입력하거나 참조 이미지를 업로드하세요!")

# ✅ 사용자 입력 받기
src_name = st.text_input("원본 이미지 이름을 입력하세요:", "sumin_face")
global_cond = st.text_input("헤어스타일 변경 조건을 입력하세요 (텍스트 또는 이미지 파일 경로):", "")

# ✅ 파일 업로드 기능
uploaded_file = st.file_uploader("참조 이미지 업로드 (선택 사항)", type=["jpg", "png"])
if uploaded_file:
    global_cond = f"./uploaded/{uploaded_file.name}"
    with open(global_cond, "wb") as f:
        f.write(uploaded_file.getbuffer())

# ✅ 편집 옵션
local_sketch = st.checkbox("로컬 스케치 사용 (사용자가 직접 수정)")
paint_the_mask = st.checkbox("헤어 마스크 조정")

# ✅ 실행 버튼
if st.button("헤어스타일 변경 실행"):
    with st.spinner("AI가 헤어스타일을 변경하는 중..."):
        result_img = hairstyle_editing_pipeline(src_name, global_cond, local_sketch, paint_the_mask)

        # ✅ 결과 이미지 표시
        st.subheader("변경된 헤어스타일")
        st.image(result_img, caption="💇‍♀️ 편집된 헤어스타일", use_column_width=True)
