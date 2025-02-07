import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import torch
import face_alignment
import numpy as np
import dlib
import torchvision
import PIL
from pathlib import Path
from PIL import Image
from torchvision import transforms
from scripts.Embedding import Embedding
from scripts.text_proxy import TextProxy
from scripts.ref_proxy import RefProxy
from scripts.sketch_proxy import SketchProxy
from scripts.bald_proxy import BaldProxy
from scripts.color_proxy import ColorProxy
from scripts.feature_blending import hairstyle_feature_blending
from utils.shape_predictor import align_face
from utils.seg_utils import vis_seg
from utils.mask_ui import painting_mask
from utils.image_utils import display_image_list, process_display_input
from utils.model_utils import load_base_models
from utils.options import Options

def alignimage():
  unaligned_dir = 'test_images/unaligned_img/'
  output_dir = 'test_images/aligned_img'
  output_size = 1024
  landmark_path = 'pretrained_models/shape_predictor_68_face_landmarks.dat'
  
  predictor = dlib.shape_predictor(landmark_path)
  
  for im in Path(unaligned_dir).glob("*.*"):
      faces = align_face(str(im), predictor)
      for i, face in enumerate(faces):
          if output_size:
              factor = 1024 // output_size
              assert output_size * factor == 1024
              face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
              face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
              face = torchvision.transforms.ToPILImage()(face_tensor_lr)
              if factor != 1:
                  face = face.resize((output_size, output_size), PIL.Image.LANCZOS)
          if len(faces) > 1:
              face.save(Path(output_dir) / (im.stem + f"_{i}.jpg"))
          else:
              face.save(Path(output_dir) / (im.stem + f".jpg"))

opts = Options().parse(jupyter=True)
src_name = 'passport'# source image name you want to edit
opts.src_img_dir = '/content/drive/MyDrive/HairCLIPv2/test_images/aligned_img'

def image_process():
  image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  g_ema, mean_latent_code, seg = load_base_models(opts)
  ii2s = Embedding(opts, g_ema, mean_latent_code[0,0])
  if not os.path.isfile(os.path.join(opts.src_latent_dir, f"{src_name}.npz")):
    inverted_latent_w_plus, inverted_latent_F = ii2s.invert_image_in_FS(image_path=f'{opts.src_img_dir}/{src_name}.jpg')
    save_latent_path = os.path.join(opts.src_latent_dir, f'{src_name}.npz')
    np.savez(save_latent_path, latent_in=inverted_latent_w_plus.detach().cpu().numpy(),
                latent_F=inverted_latent_F.detach().cpu().numpy())
  src_latent = torch.from_numpy(np.load(f'{opts.src_latent_dir}/{src_name}.npz')['latent_in']).cuda()
  src_feature = torch.from_numpy(np.load(f'{opts.src_latent_dir}/{src_name}.npz')['latent_F']).cuda()
  src_image = image_transform(Image.open(f'{opts.src_img_dir}/{src_name}.jpg').convert('RGB')).unsqueeze(0).cuda()
  input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

  bald_proxy = BaldProxy(g_ema, opts.bald_path)
  text_proxy = TextProxy(opts, g_ema, seg, mean_latent_code)
  ref_proxy = RefProxy(opts, g_ema, seg, ii2s)
  sketch_proxy = SketchProxy(g_ema, mean_latent_code, opts.sketch_path)
  color_proxy = ColorProxy(opts, g_ema, seg)
  edited_hairstyle_img = src_image

def hairstyle_editing(global_cond=None, local_sketch=False, paint_the_mask=False, \
                      src_latent=src_latent, src_feature=src_feature, input_mask=input_mask, src_image=src_image, \
                        latent_global=None, latent_local=None, latent_bald=None, local_blending_mask=None, painted_mask=None):
    if paint_the_mask:
        modified_mask = painting_mask(input_mask)
        input_mask = torch.from_numpy(modified_mask).unsqueeze(0).cuda().long().clone().detach()
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

    src_feature, edited_hairstyle_img = hairstyle_feature_blending(g_ema, seg, src_latent, src_feature, input_mask, latent_bald=latent_bald,\
                                                latent_global=latent_global, latent_local=latent_local, local_blending_mask=local_blending_mask)
    return src_feature, edited_hairstyle_img

def hairCLIPv2(hairstyle):
  alignimage()
  image_process()
  src_feature, edited_hairstyle_img = hairstyle_editing(global_cond=hairstyle, local_sketch=False, paint_the_mask=False)
  display_image_list([process_display_input(src_image), process_display_input(edited_hairstyle_img)])

def haircolor(color): #e.g. 'red hair' for text_mode; '108157.jpg' for ref_mode; (220,220,220) for RGB value mode
  visual_color_list, visual_final_list = color_proxy(color, edited_hairstyle_img, src_latent, src_feature)
  display_image_list([src_image, visual_final_list[-1]])

hairCLIPv2('bowl cut hairstyle')