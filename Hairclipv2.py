import os
import torch
import numpy as np
import streamlit as st  # ✅ Streamlit 라이브러리 추가
from PIL import Image
from torchvision import transforms
from scripts.Embedding import Embedding
from scripts.text_proxy import TextProxy
from scripts.ref_proxy import RefProxy
from scripts.sketch_proxy import SketchProxy
from scripts.bald_proxy import BaldProxy
from scripts.color_proxy import ColorProxy
from scripts.feature_blending import hairstyle_feature_blending
from utils.seg_utils import vis_seg
from utils.mask_ui import painting_mask
from utils.image_utils import display_image_list, process_display_input
from utils.model_utils import load_base_models
from utils.options import Options

# ✅ 디바이스 자동 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ✅ 이미지 변환 설정
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def hairstyle_editing_pipeline(src_name, global_cond=None, local_sketch=False, paint_the_mask=False):
    """
    헤어스타일 편집을 수행하는 함수 (Streamlit에서 진행 상황을 표시하도록 수정)
    """
    # ✅ Streamlit 진행 상태 표시 (텍스트 + 프로그레스 바)
    status_text = st.empty()
    progress_bar = st.progress(0)

    # ✅ 옵션 로드
    status_text.text("📌 옵션 로드 중...")
    opts = Options().parse(jupyter=True)
    progress_bar.progress(10)

    # ✅ 기본 모델 로드
    status_text.text("📌 모델 로딩 중...")
    g_ema, mean_latent_code, seg = load_base_models(opts)
    ii2s = Embedding(opts, g_ema, mean_latent_code[0, 0])
    progress_bar.progress(30)

    # ✅ Latent 저장 경로 확인 및 생성
    latent_path = os.path.join(opts.src_latent_dir, f"{src_name}.npz")
    if not os.path.isfile(latent_path):
        status_text.text("📌 이미지 임베딩 생성 중...")
        inverted_latent_w_plus, inverted_latent_F = ii2s.invert_image_in_FS(image_path=f'{opts.src_img_dir}/{src_name}.jpg')
        np.savez(latent_path, latent_in=inverted_latent_w_plus.detach().to(device).numpy(),
                 latent_F=inverted_latent_F.detach().to(device).numpy())
    progress_bar.progress(50)

    # ✅ Latent 로드
    status_text.text("📌 Latent 데이터 로딩 중...")
    src_latent = torch.from_numpy(np.load(latent_path)['latent_in']).to(device)
    src_feature = torch.from_numpy(np.load(latent_path)['latent_F']).to(device)
    progress_bar.progress(60)

    # ✅ 원본 이미지 및 세그멘테이션 마스크 준비
    status_text.text("📌 원본 이미지 분석 중...")
    src_image = image_transform(Image.open(f'{opts.src_img_dir}/{src_name}.jpg').convert('RGB')).unsqueeze(0).to(device)
    input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach().to(device)
    progress_bar.progress(70)

    # ✅ 프록시 생성
    status_text.text("📌 변형 모델 로딩 중...")
    bald_proxy = BaldProxy(g_ema, opts.bald_path)
    text_proxy = TextProxy(opts, g_ema, seg, mean_latent_code)
    ref_proxy = RefProxy(opts, g_ema, seg, ii2s)
    sketch_proxy = SketchProxy(g_ema, mean_latent_code, opts.sketch_path)
    color_proxy = ColorProxy(opts, g_ema, seg)
    progress_bar.progress(80)

    # ✅ 헤어스타일 편집
    status_text.text("📌 헤어스타일 변경 중...")
    latent_global, latent_local, latent_bald, local_blending_mask, painted_mask = None, None, None, None, None

    if paint_the_mask:
        modified_mask = painting_mask(input_mask)
        input_mask = torch.from_numpy(modified_mask).unsqueeze(0).to(device).long().clone().detach()
        vis_modified_mask = vis_seg(modified_mask)
        display_image_list([src_image, vis_modified_mask])
        painted_mask = input_mask

    if local_sketch:
        latent_local, local_blending_mask, visual_local_list = sketch_proxy(input_mask)
        display_image_list(visual_local_list)

    if global_cond is not None:
        assert isinstance(global_cond, str)

        latent_bald, visual_bald_list = bald_proxy(src_latent)
        display_image_list(visual_bald_list)

        if global_cond.endswith('.jpg') or global_cond.endswith('.png'):
            latent_global, visual_global_list = ref_proxy(global_cond, src_image, painted_mask=painted_mask)
        else:
            latent_global, visual_global_list = text_proxy(global_cond, src_image, from_mean=True, painted_mask=painted_mask)
        display_image_list(visual_global_list)

    # ✅ 최종 이미지 생성
    src_feature, edited_hairstyle_img = hairstyle_feature_blending(
        g_ema, seg, src_latent, src_feature, input_mask,
        latent_bald=latent_bald, latent_global=latent_global,
        latent_local=latent_local, local_blending_mask=local_blending_mask
    )

    progress_bar.progress(100)
    status_text.text("✅ 헤어스타일 변경 완료!")

    return process_display_input(edited_hairstyle_img)