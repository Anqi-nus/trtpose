import json
import argparse
import trt_pose.coco
import ipdb;

pdb = ipdb.set_trace
import trt_pose.models
import torch
import cv2
import torchvision.transforms as transforms
import PIL.Image
from tqdm import tqdm
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import time

device = torch.device('cuda')
from torch2trt import TRTModule

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


# load model. num_parts and num_links are fixed value based on the model config, do not change
def load_model(model_path):
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    print("running", model_path)
    model.load_state_dict(torch.load(model_path))
    return model.eval()

# load trt model
def load_trtmodel(model_path):
    print("running", model_path)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path))
    #  return model_trt.eval()
    return model_trt

# Pre-process the image data and resize back to size of annotation images
def preprocess(image, size):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    image = PIL.Image.fromarray(image)
    image = image.resize((size, size), resample=PIL.Image.BILINEAR)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def img_resize(image, max_length=640):
    H, W = image.shape[:2]
    if max(W, H) > max_length:  # shrink
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    if W > H:
        W_resize = max_length
        H_resize = int(H * max_length / W)
    else:
        H_resize = max_length
        W_resize = int(W * max_length / H)
    image = cv2.resize(image, (W_resize, H_resize), interpolation=interpolation)
    return image, W_resize, H_resize

# To test on an image
def img_demo(args):
    torch.cuda.current_stream().synchronize()
    model = load_model(args.model)
    image = cv2.imread(args.input)
    data = preprocess(image, args.size)
    cmap, paf = model(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    torch.cuda.current_stream().synchronize()
    counts, objects, peaks = parse_objects(cmap, paf)  # cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    cv2.imshow("torch pose estimation", image)
    cv2.waitKey(100)

# To test on a video and save the output as .mp4 file
def video_inference(args):
    cap = cv2.VideoCapture(args.input)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.trt == "trt": # by default, load trt model
        model = load_trtmodel(args.model)
    else:
        model = load_model(args.model)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type of input
    out = cv2.VideoWriter(args.output, cv2.CAP_FFMPEG,
                          fourcc, 30, (384, 288)) # store the output video in 384*288 dimension
    t0 = time.time()
    # for each frame in the input video
    for frame in tqdm(range(video_length)):
        _, image = cap.read()
        image = img_resize(image, 384)[0]
        data = preprocess(image, args.size)
        cmap, paf = model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)  # , cmap_threshold=0.15, link_threshold=0.15)
        draw_objects(image, counts, objects, peaks)
        cv2.imshow("torch pose estimation", image)
        out.write(image)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to stop
            break
    t1 = time.time()
    fps = video_length / (t1 - t0)
    print('Overall fps =', fps)
    cv2.destroyAllWindows()
    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video with trtpose".')
    parser.add_argument('--input', type=str, help='path to video to process')
    parser.add_argument('--output', type=str, help='path to video output')
    parser.add_argument('--trt', type=str, default='trt', help='Default run with trt, --trt=false to disable trt')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--size', type=int, default=256, help='size of model. e.g., densenet121_320x320 has size 320')
    args = parser.parse_args()
    video_inference(args)