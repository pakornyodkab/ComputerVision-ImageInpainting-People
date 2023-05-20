import torch
from torchvision import transforms as T
from deepfillv2.model.networks import Generator
import cv2
import random
from loguru import logger
from lama_cleaner.helper import (
    resize_max_size,
)
import time
from lama_cleaner.schema import Config
from lama_cleaner.model_manager import ModelManager
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

def inpaintByDeepfillV2(img, mask,path):
    use_cuda_if_available = False
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')
    sd_path = path
    generator = Generator(checkpoint=sd_path, return_flow=True).to(device)
    image_org = T.ToTensor()(img).to(device)
    mask = T.ToTensor()(mask).to(device)
    output = generator.infer(image_org, mask, return_vals=[
                             'inpainted', 'stage1', 'stage2', 'flow'])
    return output[0]

def inpaintByDeepfillV2Pretrain(img,mask):
    inpaintByDeepfillV2(img,mask,'states_pt_places2.pth')

def inpaintByDeepfillV2Finetune(img, mask):
    inpaintByDeepfillV2(img,mask,'states_deepfill.pth')


def diffuser_callback(i, t, latents):
    pass


def inpaintByLaMa(image, mask):
    interpolation = cv2.INTER_CUBIC
    size_limit = max(image.shape)

    config = Config(
        ldm_steps=25,
        ldm_sampler="plms",
        hd_strategy="Crop",
        zits_wireframe=True,
        hd_strategy_crop_margin=196,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=2048,
        prompt="",
        negative_prompt="",
        use_croper=False,
        croper_x=24,
        croper_y=94,
        croper_height=512,
        croper_width=512,
        sd_scale=1,
        sd_mask_blur=5,
        sd_strength=0.75,
        sd_steps=50,
        sd_guidance_scale=7.5,
        sd_sampler="uni_pc",
        sd_seed=-1,
        sd_match_histograms=False,
        cv2_flag="INPAINT_NS",
        cv2_radius=5,
        paint_by_example_steps=50,
        paint_by_example_guidance_scale=7.5,
        paint_by_example_mask_blur=5,
        paint_by_example_seed=-1,
        paint_by_example_match_histograms=False,
        paint_by_example_example_image=None,
        p2p_steps=50,
        p2p_image_guidance_scale=1.5,
        p2p_guidance_scale=7.5,
        controlnet_conditioning_scale=0.4,
    )

    LAMAModel = ModelManager(
        name="lama",
        sd_controlnet=False,
        device=torch.device('cpu'),
        no_half=False,
        hf_access_token="",
        disable_nsfw=False,
        sd_cpu_textencoder=False,
        sd_run_local=False,
        sd_local_model_path=None,
        local_files_only=False,
        cpu_offload=False,
        enable_xformers=False,
        callback=diffuser_callback,
    )

    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 999999999)
    if config.paint_by_example_seed == -1:
        config.paint_by_example_seed = random.randint(1, 999999999)
    image = resize_max_size(image, size_limit=size_limit,
                            interpolation=interpolation)
    mask = resize_max_size(mask, size_limit=size_limit,
                           interpolation=interpolation)

    start = time.time()
    print("start  to inpaint")
    res_np_img = LAMAModel(image, mask, config)
    res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return res_np_img


pixelwise_loss = torch.nn.L1Loss()

directory = './evaluateSource/image'
deepfillPretrainLosses = []
deepfillFinetuneLosses = []
lamaLosses = []
transform = transforms.ToTensor()

start = time.time()

for index,filename in enumerate(os.listdir(directory)):
    print("Index",index)
    print("Filename",filename)
    f = os.path.join(directory,filename)
    mask_f = os.path.join('./evaluateSource/mask/',filename)

    image = Image.open(f)
    image = image.resize((256,256))
    image = np.array(image)
    # image = cv2.imread(f)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    mask = Image.open(mask_f)
    mask = mask.resize((256,256))
    mask = np.array(mask)
    mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    # mask = cv2.imread(mask_f)
    # mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)

    # deepfillPretrain = inpaintByDeepfillV2Pretrain(image,mask)
    # deepfillFinetune = inpaintByDeepfillV2Finetune(image,mask)
    lama = inpaintByLaMa(image,mask)

    # print("Lama type",type(lama.dtype))
    # print("Image type",type(image.dtype))

    # deepfillPretrainLoss = pixelwise_loss(deepfillPretrain , image)
    # deepfillFinetuneLoss = pixelwise_loss(deepfillFinetune , image)
    lamaLoss = pixelwise_loss(transform(lama),transform(image))

    # deepfillPretrainLosses.append(deepfillPretrainLoss)
    # deepfillFinetuneLosses.append(deepfillFinetuneLoss)
    lamaLosses.append(lamaLoss)

end = time.time()

# print("DeepfillV2 Pretrained loss",sum(deepfillPretrainLosses)/len(deepfillPretrainLosses))
# print("DeepfillV2 Fine tune loss",sum(deepfillFinetuneLosses)/len(deepfillFinetuneLosses))
print("Lama Loss",sum(lamaLosses)/len(lamaLosses))

print("Time use",end-start)



