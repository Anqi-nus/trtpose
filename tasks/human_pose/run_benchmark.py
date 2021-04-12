import torch
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import json
import trt_pose.coco
from torch2trt import TRTModule
from jetcam.utils import bgr8_to_jpeg
import ipywidgets
from IPython.display import display

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)

WIDTH = 320
HEIGHT = 320
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

OPTIMIZED_MODEL = 'densenet121_baseline_att_320x320_A_epoch_10_trt.pth'
print('Loading trt model',OPTIMIZED_MODEL,'...') 
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
    

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

print('Running inference...')
image = cv2.imread('/home/conex/Downloads/test/frame_18.jpg')
data = preprocess(image)
cmap, paf = model_trt(data)
cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
torch.cuda.current_stream().synchronize()

counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
draw_objects(image, counts, objects, peaks)
cv2.imshow("torch pose estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(counts, objects, peaks)
'''
print('image_w:')
print('value =',image_w)
print('type =', type(image_w))


image_w = ipywidgets.Image(format='jpeg')
display(image_w)

image = cv2.imread('/home/conex/Downloads/test/frame_18.jpg')
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cmap, paf = model_trt(data)
cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
draw_objects(image, counts, objects, peaks)
image_w = bgr8_to_jpeg(image[:, ::-1, :])
'''

