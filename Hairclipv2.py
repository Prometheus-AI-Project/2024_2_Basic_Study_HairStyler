import os
import torch
import argparse
import numpy as np
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


# ✅ 이미지 변환을 위한 기본 설정
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def hairstyle_editing_pipeline(src_name, device, global_cond=None, local_sketch=False, paint_the_mask=False):
    """
    헤어스타일 편집을 수행하는 함수.
    - src_name: 원본 이미지 파일명 (예: "sumin_face")
    - global_cond: 헤어스타일 변경 조건 (텍스트 또는 이미지 경로)
    - local_sketch: 사용자가 직접 스케치를 적용할지 여부 (True / False)
    - paint_the_mask: 헤어 마스크를 조정할지 여부 (True / False)
    """
    # ✅ 옵션 로드
    opts = Options().parse(jupyter=True)

    # ✅ 기본 모델 로드
    g_ema, mean_latent_code, seg = load_base_models(opts)
    ii2s = Embedding(opts, g_ema, mean_latent_code[0, 0])

    # ✅ Latent 저장 경로 확인 및 생성
    latent_path = os.path.join(opts.src_latent_dir, f"{src_name}.npz")
    if not os.path.isfile(latent_path):
        inverted_latent_w_plus, inverted_latent_F = ii2s.invert_image_in_FS(image_path=f'{opts.src_img_dir}/{src_name}.jpg')
        np.savez(latent_path, latent_in=inverted_latent_w_plus.detach().cpu().numpy(),
                 latent_F=inverted_latent_F.detach().cpu().numpy())

    # ✅ Latent 로드
    src_latent = torch.from_numpy(np.load(latent_path)['latent_in']).to(device)
    src_feature = torch.from_numpy(np.load(latent_path)['latent_F']).to(device)

    # ✅ 원본 이미지 및 세그멘테이션 마스크 준비
    src_image = image_transform(Image.open(f'{opts.src_img_dir}/{src_name}.jpg').convert('RGB')).unsqueeze(0).to(device)
    input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

    # ✅ 프록시 생성 (헤어스타일 변형용 모듈)
    bald_proxy = BaldProxy(g_ema, opts.bald_path)
    text_proxy = TextProxy(opts, g_ema, seg, mean_latent_code)
    ref_proxy = RefProxy(opts, g_ema, seg, ii2s)
    sketch_proxy = SketchProxy(g_ema, mean_latent_code, opts.sketch_path)
    color_proxy = ColorProxy(opts, g_ema, seg)

    # ✅ 헤어스타일 편집 파이프라인
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

        # 대머리 변환
        latent_bald, visual_bald_list = bald_proxy(src_latent)
        display_image_list(visual_bald_list)

        # 참조 이미지 기반 편집
        if global_cond.endswith('.jpg') or global_cond.endswith('.png'):
            latent_global, visual_global_list = ref_proxy(global_cond, src_image, painted_mask=painted_mask)
        else:
            # 텍스트 기반 편집
            latent_global, visual_global_list = text_proxy(global_cond, src_image, from_mean=True, painted_mask=painted_mask)
        display_image_list(visual_global_list)

    # ✅ 최종 헤어스타일 편집
    src_feature, edited_hairstyle_img = hairstyle_feature_blending(
        g_ema, seg, src_latent, src_feature, input_mask,
        latent_bald=latent_bald, latent_global=latent_global,
        latent_local=latent_local, local_blending_mask=local_blending_mask
    )

    return process_display_input(edited_hairstyle_img)


### ✅ **메인 실행 함수**
def main():
    parser = argparse.ArgumentParser(description="Hair Style Editing AI")
    parser.add_argument("src_name", type=str, help="원본 이미지 파일명 (예: sumin_face)")
    parser.add_argument("--global_cond", type=str, default=None, help="헤어스타일 변경 조건 (텍스트 또는 이미지 경로)")
    parser.add_argument("--local_sketch", action="store_true", help="로컬 스케치 적용 여부")
    parser.add_argument("--paint_the_mask", action="store_true", help="헤어 마스크 조정 여부")

    # 자동으로 CUDA, MPS, CPU 중 사용 가능한 디바이스 선택
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac용 Metal API
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    else:
        device = torch.device("cpu")   # 기본 CPU

    args = parser.parse_args()

    # ✅ 실행
    result_img = hairstyle_editing_pipeline(
        args.src_name, device, args.global_cond, args.local_sketch, args.paint_the_mask
    )

    # ✅ 결과 출력
    result_img.show()  # PIL 이미지 직접 열기
    result_img.save(f"./results/{args.src_name}_edited.jpg")  # 결과 저장
    print(f"✅ 헤어스타일 변경 완료! 결과 저장: ./results/{args.src_name}_edited.jpg")


# ✅ **직접 실행할 경우 `main()` 호출**
if __name__ == "__main__":
    main()