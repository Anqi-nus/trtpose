import json
import argparse
import trt_pose.coco
import ipdb; pdb=ipdb.set_trace
import trt_pose.models
import torch
import cv2
#import torch2trt
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

def load_model():
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    MODEL_WEIGHTS = '/home/conex/Downloads/epoch_120.pth'
    print("running", MODEL_WEIGHTS)
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    return model.eval()


# load trt model
def load_trtmodel():
    OPTIMIZED_MODEL = './models/densenet169_256x256_epoch130_trt.pth'
    print("running", OPTIMIZED_MODEL)
    
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
    #  return model_trt.eval()
    return model_trt

def preprocess(image):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    image = PIL.Image.fromarray(image)
    image = image.resize((256,256),resample=PIL.Image.BILINEAR)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def img_resize(image, max_length=640):
    H, W = image.shape[:2]
    if max(W, H) > max_length: #shrink
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    if W>H:
        W_resize = max_length
        H_resize = int(H * max_length / W)
    else:
        H_resize = max_length
        W_resize = int(W * max_length / H)
    image = cv2.resize(image, (W_resize, H_resize), interpolation=interpolation)
    return image, W_resize, H_resize


def img_demo(path):
    torch.cuda.current_stream().synchronize()
    model = load_model()
    image = cv2.imread(path)
    data = preprocess(image)
    cmap, paf = model(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    torch.cuda.current_stream().synchronize()
    counts, objects, peaks = parse_objects(cmap, paf) # cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    cv2.imshow("torch pose estimation", image)
    cv2.waitKey(100)

def video_infrence(video_name, output_name):
    cap = cv2.VideoCapture(video_name) 
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #model = load_trtmodel()
    model = load_model()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, cv2.CAP_FFMPEG, fourcc, 30, (640, 480))
    t0 = time.time()
    for i in tqdm(range(video_length)):
        _, image = cap.read()
        image = img_resize(image, 640)[0]
        data = preprocess(image)
        cmap, paf = model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        draw_objects(image, counts, objects, peaks)
        cv2.imshow("torch pose estimation", image)
        out.write(image)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to stop
            break
    t1 = time.time()
    fps = video_length/(t1-t0)
    print('fps =', fps)
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video with trtpose".')
    parser.add_argument('input', type=str, help='The path of video to process')
    parser.add_argument('output', type=str, help='Name of output video')
    args = parser.parse_args()
    video_infrence(args.input, args.output, args.trt)
