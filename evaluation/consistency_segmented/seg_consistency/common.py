"""Shared frame loading / encoders for the segmentation-enhanced consistency metrics."""
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

try:
    BICUBIC = transforms.InterpolationMode.BICUBIC
except AttributeError:
    BICUBIC = Image.BICUBIC

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def load_frames_rgb(video_path):
    """Return list of HxWx3 uint8 RGB numpy frames (ALL frames)."""
    if os.path.isdir(video_path):
        exts = ('jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff',
                'JPG', 'PNG', 'JPEG', 'BMP', 'TIF', 'TIFF')
        imgs = sorted([p for p in glob.glob(os.path.join(video_path, '*'))
                       if os.path.splitext(p)[1][1:] in exts])
        frames = [np.array(Image.open(p).convert('RGB')).astype(np.uint8) for p in imgs]
        assert frames, f"No frames in {video_path}"
        return frames
    elif video_path.endswith('.mp4'):
        from decord import VideoReader
        import decord
        decord.bridge.set_bridge('native')
        vr = VideoReader(video_path, num_threads=1)
        arr = vr.get_batch(range(len(vr))).asnumpy().astype(np.uint8)  # (T,H,W,C)
        return [arr[i] for i in range(arr.shape[0])]
    raise NotImplementedError(f"Unsupported input: {video_path}")


def collect_videos(videos_path):
    if os.path.isfile(videos_path) and videos_path.endswith('.mp4'):
        return [videos_path]
    videos = sorted(glob.glob(os.path.join(videos_path, '*.mp4')))
    if videos:
        return videos
    return sorted([d for d in glob.glob(os.path.join(videos_path, '*')) if os.path.isdir(d)])


def _preproc(np_rgb, n_px, mean, std, device):
    """np uint8 HxWx3 -> normalized (1,3,n,n) tensor."""
    t = torch.from_numpy(np_rgb).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
    t = F.interpolate(t, size=(n_px, n_px), mode='bicubic', align_corners=False, antialias=True)
    t = t.div(255.0)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    t = (t - m) / s
    return t.to(device)


# ---------------- Encoders ----------------
class DinoV2Encoder:
    """DINOv2 ViT-B/14 (default) — stronger successor of DINO, for SUBJECT."""
    def __init__(self, device, variant='dinov2_vitb14'):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', variant).to(device).eval()

    @torch.no_grad()
    def encode(self, np_rgb):
        x = _preproc(np_rgb, 224, IMNET_MEAN, IMNET_STD, self.device)
        feat = self.model(x)               # (1, D)
        return F.normalize(feat, dim=-1, p=2)


class DinoV1Encoder:
    """Original DINO ViT-B/16 — for exact parity with VBench subject metric."""
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device).eval()

    @torch.no_grad()
    def encode(self, np_rgb):
        x = _preproc(np_rgb, 224, IMNET_MEAN, IMNET_STD, self.device)
        feat = self.model(x)
        return F.normalize(feat, dim=-1, p=2)


class ClipEncoder:
    """CLIP image encoder — for BACKGROUND. Default OpenAI ViT-B/32 (VBench parity);
    pass model_name='ViT-L/14' for a stronger variant."""
    def __init__(self, device, model_name='ViT-B/32'):
        import clip
        self.device = device
        self.model, _ = clip.load(model_name, device=device)
        self.n_px = self.model.visual.input_resolution if hasattr(self.model.visual, 'input_resolution') else 224

    @torch.no_grad()
    def encode(self, np_rgb):
        x = _preproc(np_rgb, self.n_px, CLIP_MEAN, CLIP_STD, self.device)
        feat = self.model.encode_image(x)
        return F.normalize(feat.float(), dim=-1, p=2)


def cosine(a, b):
    return max(0.0, F.cosine_similarity(a, b).item())
