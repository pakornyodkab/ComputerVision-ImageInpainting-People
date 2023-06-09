import tkinter as tk
from tkinter import filedialog
import torchvision
import torch
from PIL import Image, ImageTk, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms as T
import git
from deepfillv2.model.networks import Generator
from scipy import stats as st
image_path = ""
marked_dots = []

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
count = 0
kernelSize = 5
# git.Git("/partialconv").clone("https://github.com/pakornyodkab/partialconv")
import random
from loguru import logger
from lama_cleaner.helper import (
    resize_max_size,
)
import time
from lama_cleaner.schema import Config
from lama_cleaner.model_manager import ModelManager

COCO_INSTANCE_CATEGORY_NAMES = [
  '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
  'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
  'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
  'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
 
def get_prediction(img_path, threshold):
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return masks, pred_boxes, pred_class

def browse_image():
  global marked_dots
  marked_dots = []
  global image_path
  image_path = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image Files", "*.jpg *.png *.jpeg"),))
  img = Image.open(image_path)
  img_tk = ImageTk.PhotoImage(img)
  panel.configure(image=img_tk)
  panel.image = img_tk
    

def get_mask_labels(img, human_mask, bboxes):

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  ret2,th2 = cv2.threshold(img[:,:,2],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  mask_out = np.where(human_mask > 0,  0, 1)
  th2 = th2 * mask_out
  th2 = np.uint8(th2)
  analyze = cv2.connectedComponentsWithStats(th2,4,cv2.CV_32S)

  human_c_y, human_c_x = int((bboxes[0][1]+bboxes[1][1])/2), int((bboxes[0][0]+bboxes[1][0])/2)
  (totalLabels, label_ids, values, centroid) = analyze
  for i in range(len(centroid)):
    y, x = int(centroid[i][1]), int(centroid[i][0])
    if y < human_c_y:
      mask_out = np.where(label_ids == i,  0, 1)
      th2 = th2 * mask_out

  for i in range(th2.shape[0]):
    for j in range(th2.shape[1]):
      if human_mask[i][j] > 0:
        th2[i][j] = 255

  th2 = np.uint8(th2)
  analysis = cv2.connectedComponentsWithStats(th2,4,cv2.CV_32S)
  (totalLabels, label_ids, values, centroid) = analysis
  new_human_mask = np.where(human_mask > 0, 1, 0)
  labels = []
  for i in range(new_human_mask.shape[0]):
    for j in range(new_human_mask.shape[1]):
      if new_human_mask[i][j] > 0:
        labels.append(label_ids[i][j])
  mode = st.mode(labels).mode[0]
  human_area = np.sum(np.concatenate(new_human_mask))
  if mode != 0 and values[mode][-1] < 2 * human_area:
    final_mask = np.array(np.where(label_ids == mode, 255, 0), dtype = np.uint8)
    for i in range(new_human_mask.shape[0]):
      for j in range(new_human_mask.shape[1]):
        if new_human_mask[i][j] > 0:
          final_mask[i][j] = 255
  else:
    final_mask = np.where(human_mask > 0, 255, 0)
  return final_mask

def instance_segmentation_api(img_path, marked_dots, threshold=0.5, rect_th=3, text_size=3, text_th=3):
  masks, boxes, pred_cls = get_prediction(img_path, threshold)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  h, w = img.shape[:2]
  merged_mask = np.zeros((h, w), dtype=np.uint8)
  merged_box = np.zeros((h, w), dtype=np.uint8)
  for i in range(len(masks)):
    for dot in marked_dots:
      if masks[i][dot[1],dot[0]] != 0:
        final_mask = get_mask_labels(img,masks[i],boxes[i])
        img[final_mask != 0] = [255,255,255]
        merged_mask[final_mask != 0] = 255
        merged_box[round(boxes[i][0][1]):round(boxes[i][1][1]),round(boxes[i][0][0]):round(boxes[i][1][0])] = 255
  return merged_mask, img ,merged_box

def dilateImage(pillowImage):
  #cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE
  kernelShape = cv2.MORPH_RECT
  opencvImage = cv2.cvtColor(np.array(pillowImage), cv2.COLOR_RGB2BGR)
  kernel = cv2.getStructuringElement(kernelShape,(kernelSize,kernelSize))
  outputImage = cv2.dilate(opencvImage, kernel, iterations=5)
  return Image.fromarray(cv2.cvtColor(outputImage, cv2.COLOR_BGR2GRAY))

def mergeImageWithDilateMask(pillowImage,pillowDilateMask):
  imageArray = np.array(pillowImage)
  dilateMaskArray = np.array(pillowDilateMask)
  imageArray[dilateMaskArray != 0] = [255, 255, 255]
  return Image.fromarray(imageArray)

def mark_dot(event):
  marked_dots.append((event.x, event.y))
  setState(2)
  display_image_with_dots()

def masking():
  img = Image.open(image_path)
  merged_mask, imgWhited,merged_box = instance_segmentation_api(image_path, marked_dots)
  merged_maskImage = Image.fromarray(merged_mask)
  # merged_maskImage = ImageOps.invert(merged_maskImage)
  imgWhitedImage = Image.fromarray(imgWhited)
  merge_boxImage = Image.fromarray(merged_box)
  imgWhitedImage.save("whited.jpg")
  merged_maskImage.save("mask.jpg")
  merge_boxImage.save("box.jpg")
  merged_maskImage_dilate = dilateImage(merged_maskImage)
  merged_maskImage_dilate.save("mask_dilate.jpg")

  mergedImageWithDilateMask = mergeImageWithDilateMask(img,merged_maskImage_dilate)
  mergedImageWithDilateMask.save("whitedWithDilate.jpg")
  img = Image.open("whitedWithDilate.jpg")
  img_tk = ImageTk.PhotoImage(img)
  panel.configure(image=img_tk)
  panel.image = img_tk
  setState(3)

def diffuser_callback(i, t, latents):
  pass
  # socketio.emit('diffusion_step', {'diffusion_step': step})

def predict(image,mask):
  image = np.array(image)
  mask = np.array(mask)

  original_shape = image.shape
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

  # LAMAModel = ModelManager(
  #   name="lama",
  #   device=torch.device('cpu')
  # )


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

  logger.info(f"Origin image shape: {original_shape}")
  image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)
  logger.info(f"Resized image shape: {image.shape}")

  mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

  start = time.time()
  print("start  to inpaint")
  try:
    res_np_img = LAMAModel(image, mask, config)
  except RuntimeError as e:
    torch.cuda.empty_cache()
    if "CUDA out of memory. " in str(e):
      # NOTE: the string may change?
      return None
    else:
      logger.exception(e)
      return None
  finally:
    logger.info(f"process time: {(time.time() - start) * 1000}ms")
    torch.cuda.empty_cache()

  res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
  return Image.fromarray(res_np_img)

def inpainting():
  global count
  count += 1
  img = Image.open("whitedWithDilate.jpg")
  mask_dilate = cv2.imread('mask_dilate.jpg')
  mask_dilate = cv2.cvtColor(mask_dilate,cv2.COLOR_BGR2RGB)
  use_cuda_if_available = False
  device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
  sd_path = 'states_pt_places2.pth'
  generator = Generator(checkpoint=sd_path, return_flow=True).to(device)
  image_org = T.ToTensor()(img).to(device)
  mask = T.ToTensor()(mask_dilate).to(device)
  output = generator.infer(image_org, mask, return_vals=['inpainted', 'stage1', 'stage2', 'flow'])
  imgWhitedImage = Image.fromarray(output[0])
  imgWhitedImage.save(f"result_{count}.jpg")

  img = Image.open(f"result_{count}.jpg")
  img_tk = ImageTk.PhotoImage(img)
  panel.configure(image=img_tk)
  panel.image = img_tk
  setState(4)

def inpaintingByLama():
  global count
  count += 1
  img = Image.open(image_path)
  mask_dilate = Image.open("mask_dilate.jpg")
  imgWhitedImage = predict(img,mask_dilate)
  imgWhitedImage.save(f"result_{count}.jpg")

  img = Image.open(f"result_{count}.jpg")
  img_tk = ImageTk.PhotoImage(img)
  panel.configure(image=img_tk)
  panel.image = img_tk
  setState(5)

def reInpainting():
  global count
  image = Image.open(f"result_{count}.jpg")
  mask = Image.open("mask_dilate.jpg")
  mask = mask.convert('RGB')
  mask = mask.resize(image.size)
  inv_mask = ImageOps.invert(mask)
  result = Image.composite(image, Image.new('RGB', image.size, (255, 255, 255)), inv_mask)
  count += 1
  use_cuda_if_available = True
  device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
  sd_path = 'states_pt_places2.pth'
  generator = Generator(checkpoint=sd_path, return_flow=True).to(device)
  image_org = T.ToTensor()(result).to(device)
  mask = T.ToTensor()(mask).to(device)
  output = generator.infer(image_org, mask, return_vals=['inpainted', 'stage1', 'stage2', 'flow'])
  imgWhitedImage = Image.fromarray(output[0])
  imgWhitedImage.save(f"result_{count}.jpg")

  img = Image.open(f"result_{count}.jpg")
  img_tk = ImageTk.PhotoImage(img)
  panel.configure(image=img_tk)
  panel.image = img_tk

def reInpaintingByLama():
  global count
  img = Image.open(f"result_{count}.jpg")
  mask_dilate = Image.open("mask_dilate.jpg")
  count += 1

  imgWhitedImage = predict(img,mask_dilate)
  imgWhitedImage.save(f"result_{count}.jpg")

  img = Image.open(f"result_{count}.jpg")
  img_tk = ImageTk.PhotoImage(img)
  panel.configure(image=img_tk)
  panel.image = img_tk

def clear():
  global marked_dots
  marked_dots = []
  global image_path
  image_path = ''
  # display_image_with_dots()
  panel.configure(image='')
  setState(1)

def display_image_with_dots():
  img = Image.open(image_path)
  draw = ImageDraw.Draw(img)
  radius = 2
  for dot in marked_dots:
    x, y = dot
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="white")
  img_tk = ImageTk.PhotoImage(img)
  panel.configure(image=img_tk)
  panel.image = img_tk

def update_slice_value(val):
  global kernelSize
  kernelSize = int(val)

def download_image():
  img = Image.open(f"result_{count}.jpg")
  img.save("save-result.jpg")

def setState(state):
  if state == 1:
    browse_button.pack()
    slice_bar.pack_forget()
    mask_button.pack_forget()
    inpainting_button.pack_forget()
    inpainting_by_lama_button.pack_forget()
    clear_button.pack()
    reinpainting_button.pack_forget()
    reinpainting_by_lama_button.pack_forget()
    download_button.pack_forget()
  elif state == 2:
    browse_button.pack_forget()
    slice_bar.pack()
    mask_button.pack()
    inpainting_button.pack_forget()
    inpainting_by_lama_button.pack_forget()
    clear_button.pack()
    reinpainting_button.pack_forget()
    reinpainting_by_lama_button.pack_forget()
    download_button.pack_forget()
  elif state == 3:
    browse_button.pack_forget()
    slice_bar.pack_forget()
    mask_button.pack_forget()
    inpainting_button.pack()
    inpainting_by_lama_button.pack()
    clear_button.pack()
    reinpainting_button.pack_forget()
    reinpainting_by_lama_button.pack_forget()
    download_button.pack_forget()
  elif state == 4:
    browse_button.pack_forget()
    slice_bar.pack_forget()
    mask_button.pack_forget()
    inpainting_button.pack_forget()
    inpainting_by_lama_button.pack_forget()
    clear_button.pack()
    reinpainting_button.pack()
    reinpainting_by_lama_button.pack_forget()
    download_button.pack()
  elif state == 5:
    browse_button.pack_forget()
    slice_bar.pack_forget()
    mask_button.pack_forget()
    inpainting_button.pack_forget()
    inpainting_by_lama_button.pack_forget()
    clear_button.pack()
    reinpainting_button.pack_forget()
    reinpainting_by_lama_button.pack()
    download_button.pack()
root = tk.Tk()
root.attributes('-fullscreen', False)

canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(root, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

panel = tk.Label(frame)
panel.pack()
panel.bind("<Button-1>", mark_dot)

browse_button = tk.Button(frame, text="Browse", command=browse_image)
browse_button.pack()

slice_bar = tk.Scale(frame, from_=1, to=9, orient=tk.HORIZONTAL, length=200, command=update_slice_value)
slice_bar.set(kernelSize)
slice_bar.pack()

mask_button = tk.Button(frame, text="Mask", command=masking)
mask_button.pack()

inpainting_button = tk.Button(frame, text="Inpainting By Deepfill", command=inpainting)
inpainting_button.pack()

inpainting_by_lama_button = tk.Button(frame, text="Inpainting By Lama", command=inpaintingByLama)
inpainting_by_lama_button.pack()

clear_button = tk.Button(frame, text="Clear", command=clear)
clear_button.pack()

reinpainting_button = tk.Button(frame, text="Re-Inpainting By Deepfill", command=reInpainting)
reinpainting_button.pack()

reinpainting_by_lama_button = tk.Button(frame, text="Re-Inpainting by Lama", command=reInpaintingByLama)
reinpainting_by_lama_button.pack()

download_button = tk.Button(frame, text="Download Image", command=download_image)
download_button.pack()

def update_canvas_scrollregion(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", update_canvas_scrollregion)
setState(1)
root.mainloop()