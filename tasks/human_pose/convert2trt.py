#!/usr/bin/env python
# coding: utf-8

import torch
import torch2trt
from torch2trt import TRTModule
import json
import trt_pose.coco
import trt_pose.models
import time

print("Loading topology...")
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

print("Loading model backbone...")
model = trt_pose.models.densenet169_baseline_att(num_parts, 2 * num_links).cuda().eval()

print("Loading model weight...")

# path to model to convert
MODEL_WEIGHTS = './models/densenet169_256x256_epoch130.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

print("Generating data...")
# change to model width and height
WIDTH =256
HEIGHT=256
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
print("Start converting...")

# set fp16 precision, and workspace size when converting
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<24)

print("Loading trt model...")
# change to designated trt model path
OPTIMIZED_MODEL = './models/densenet169_256x256_epoch130_trt.pth'

print("Saving trt model in",OPTIMIZED_MODEL)
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL																																																																																																																																			)
print("Running trt benchmark...")
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))
