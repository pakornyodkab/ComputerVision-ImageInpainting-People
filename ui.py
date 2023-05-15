import tkinter as tk
from tkinter import filedialog
import torchvision
import torch
from PIL import Image, ImageTk, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms as T
# import git
from deepfillv2.model.networks import Generator

image_path = ""
marked_dots = []

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
count = 0
# git.Git("/deepfillv2").clone("https://github.com/NATCHANONPAN/deepfillv2.git")

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
  global image_path
  image_path = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image Files", "*.jpg *.png *.jpeg"),))
  img = Image.open(image_path)
  img_tk = ImageTk.PhotoImage(img)
  panel.configure(image=img_tk)
  panel.image = img_tk
    
def instance_segmentation_api(img_path, marked_dots, threshold=0.5, rect_th=3, text_size=3, text_th=3):
  masks, boxes, pred_cls = get_prediction(img_path, threshold)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  h, w = img.shape[:2]
  merged_mask = np.zeros((h, w), dtype=np.uint8)
  for i in range(len(masks)):
    for dot in marked_dots:
      if masks[i][dot[1],dot[0]] != 0:
        img[masks[i] != 0] = [255,255,255]
        merged_mask[masks[i] != 0] = 255
  return merged_mask, img

def dilateImage(pillowImage):
  #cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE
  kernelShape = cv2.MORPH_RECT
  kernelSize = 5
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
  display_image_with_dots()

def masking():
  img = Image.open(image_path)
  merged_mask, imgWhited = instance_segmentation_api(image_path, marked_dots)
  merged_maskImage = Image.fromarray(merged_mask)
  imgWhitedImage = Image.fromarray(imgWhited)
  imgWhitedImage.save("whited.jpg")
  merged_maskImage.save("mask.jpg")
  merged_maskImage_dilate = dilateImage(merged_maskImage)
  merged_maskImage_dilate.save("mask_dilate.jpg")

  mergedImageWithDilateMask = mergeImageWithDilateMask(img,merged_maskImage_dilate)
  mergedImageWithDilateMask.save("whitedWithDilate.jpg")
  img = Image.open("whitedWithDilate.jpg")
  img_tk = ImageTk.PhotoImage(img)
  panel.configure(image=img_tk)
  panel.image = img_tk  

def inpainting():
  print("in")
  global count
  count += 1
  img = Image.open("whitedWithDilate.jpg")
  mask_dilate = Image.open("mask_dilate.jpg")

  use_cuda_if_available = True
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

def reInpainting():
  global count
  image = Image.open(f"result_{count}.jpg")
  mask = Image.open("mask_dilate.jpg")
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

def clear_dots():
  global marked_dots
  marked_dots = []
  display_image_with_dots()

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

root = tk.Tk()
root.attributes('-fullscreen', True)

panel = tk.Label(root)
panel.pack()

browse_button = tk.Button(root, text="Browse", command=browse_image)
browse_button.pack()

panel.bind("<Button-1>", mark_dot)

run_button = tk.Button(root, text="Mask", command=masking)
run_button.pack()

run_button = tk.Button(root, text="Inpainting", command=inpainting)
run_button.pack()

run_button = tk.Button(root, text="Re-Inpainting", command=reInpainting)
run_button.pack()

clear_button = tk.Button(root, text="Clear Dots", command=clear_dots)
clear_button.pack()

root.mainloop()
