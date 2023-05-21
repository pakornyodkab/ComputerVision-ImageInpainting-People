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
import argparse
from partialconv.predict import predict
import copy

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
    return inpaintByDeepfillV2(img,mask,'states_pt_places2.pth')

def inpaintByDeepfillV2Finetune(img, mask):
    return inpaintByDeepfillV2(img,mask,'states_deepfill.pth')


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


pixelwise_loss = torch.nn.L1Loss(reduction='sum')

directory = './evaluateSource/image'
deepfillPretrainLosses = []
deepfillFinetuneLosses = []
lamaLosses = []
partialConvoLosses = []
transform = transforms.ToTensor()
h=256
w=256

start = time.time()

for index,filename in enumerate(os.listdir(directory)):
    print("Index",index)
    print("Filename",filename)
    f = os.path.join(directory,filename)
    mask_f = os.path.join('./evaluateSource/mask/',filename)
    mask_black_f = os.path.join('./evaluateSource/maskBlack/',filename)
    pretrainPartivalConvoPath = os.path.join('./partialconv/','pretrained_pconv.pth')

    image = Image.open(f)
    image = image.resize((h,w))
    image = np.array(image)
    # image = cv2.imread(f)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    mask = Image.open(mask_f)
    mask = mask.resize((h,w))
    mask = np.array(mask)
    mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    maskDeepfill = cv2.imread(mask_f)
    maskDeepfill = cv2.cvtColor(maskDeepfill,cv2.COLOR_BGR2RGB)

    deepfillPretrain = inpaintByDeepfillV2Pretrain(image,maskDeepfill)
    deepfillFinetune = inpaintByDeepfillV2Finetune(image,maskDeepfill)
    lama = inpaintByLaMa(image,mask)
    args = argparse.Namespace(img=f, mask=mask_black_f,model=pretrainPartivalConvoPath, resize=True, gpu_id=0)
    partialConvoImage = predict(args)
    partialConvoImage = Image.fromarray(partialConvoImage).resize((h,w))
    partialConvoImage = np.array(partialConvoImage)

    deepfillPretrainLoss = pixelwise_loss(transform(deepfillPretrain) , transform(image))
    deepfillFinetuneLoss = pixelwise_loss(transform(deepfillFinetune) , transform(image))
    lamaLoss = pixelwise_loss(transform(lama),transform(image))
    partialConvoLoss = pixelwise_loss(transform(partialConvoImage),transform(image))

    deepfillPretrainLosses.append(deepfillPretrainLoss.item()/np.sum(np.where(mask == 255,1,0)))
    deepfillFinetuneLosses.append(deepfillFinetuneLoss.item()/np.sum(np.where(mask == 255,1,0)))
    lamaLosses.append(lamaLoss.item()/np.sum(np.where(mask == 255,1,0)))
    partialConvoLosses.append(partialConvoLoss.item()/np.sum(np.where(mask == 255,1,0)))


end = time.time()

print("DeepfillV2 Pretrained loss",sum(deepfillPretrainLosses)/len(deepfillPretrainLosses))
print("DeepfillV2 Fine tune loss",sum(deepfillFinetuneLosses)/len(deepfillFinetuneLosses))
print("Lama Loss",sum(lamaLosses)/len(lamaLosses))
print("Partial Convolution Loss",sum(partialConvoLosses)/len(partialConvoLosses))

print("Time use",end-start)



