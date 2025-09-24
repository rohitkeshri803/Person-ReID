
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import math
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T

# ---- Config ----
DATASET_BASE = os.environ.get("MARKET1501_DIR", "Market-1501-v15.09.15")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", "")  # optional path to your fine-tuned weights

# ---- Import your feature extractor ----
# Expecting a class `model(nn.Module)` that returns a pooled 2048-d feature per image
from feature_extractor import model as FeatureExtractor


# ---- Market-1501 utils ----
def _parse_market1501_filename(fname: str) -> Tuple[int, int]:
    """
    Parse Market-1501 filename like: 0002_c1s1_000451_03.jpg
    Returns (pid, camid). pid == -1 indicates junk images.
    """
    name = os.path.basename(fname)
    pid = int(name.split("_")[0])
    camid = int(name.split("_")[1][1]) - 1  # c1 -> 0-based
    return pid, camid


def _list_images(root: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(r, f))
    out.sort()
    return out


class ImgDataset(Dataset):
    def __init__(self, img_paths: List[str], transform):
        self.imgs = img_paths
        self.transform = transform
        self.pids = []
        self.camids = []
        for p in self.imgs:
            pid, camid = _parse_market1501_filename(p)
            self.pids.append(pid)
            self.camids.append(camid)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        return self.transform(img), self.pids[idx], self.camids[idx]


# ---- Metrics (CMC & mAP) ----
def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [N, D], b: [M, D] -> dist: [N, M]
    a = nn.functional.normalize(a, dim=1)
    b = nn.functional.normalize(b, dim=1)
    sim = a @ b.t()
    return 1.0 - sim  # cosine distance


def evaluate(query_feats, query_pids, query_camids, gallery_feats, gallery_pids, gallery_camids, topk=(1,5,10)):
    """
    Standard Market-1501 evaluation:
    - For each query, remove gallery samples that have the same pid & camid.
    - Compute CMC and AP, then average to get mAP.
    """
    qf = torch.from_numpy(query_feats)
    gf = torch.from_numpy(gallery_feats)
    dist = _cosine_distance(qf, gf).cpu().numpy()  # [num_q, num_g]

    num_q, num_g = dist.shape
    cmc = np.zeros(num_g, dtype=np.float64)
    aps = []

    for i in range(num_q):
        q_pid = query_pids[i]
        q_cam = query_camids[i]

        # filter: valid = not same pid & cam
        order = np.argsort(dist[i])  # ascending (smaller = more similar)
        remove = (gallery_pids == q_pid) & (gallery_camids == q_cam)
        keep = np.invert(remove)

        # good & junk sets
        good = (gallery_pids == q_pid) & (gallery_camids != q_cam)
        good_idx = np.where(good)[0]
        if len(good_idx) == 0:
            # no valid positives in gallery for this query
            continue

        # ranks after filtering
        order = order[keep[order]]
        # binary hits
        hits = (gallery_pids[order] == q_pid).astype(np.int32)

        # CMC
        hit_positions = np.where(hits == 1)[0]
        if len(hit_positions) == 0:
            continue
        cmc[hit_positions[0]:] += 1

        # AP
        # precision at each correct hit
        precisions = []
        correct = 0
        for rank, h in enumerate(hits, start=1):
            if h:
                correct += 1
                precisions.append(correct / rank)
        aps.append(np.mean(precisions) if precisions else 0.0)

    cmc = cmc / max(1, num_q)
    mAP = float(np.mean(aps)) if aps else 0.0

    results = {"mAP": mAP}
    for k in topk:
        if k <= cmc.shape[0]:
            results[f"Rank-{k}"] = float(cmc[k-1])
    return results


# ---- Feature extraction ----
def extract_features(model, loader):
    model.eval()
    feats, pids, camids = [], [], []
    with torch.no_grad():
        for imgs, batch_pids, batch_camids in loader:
            imgs = imgs.to(DEVICE)
            f = model(imgs)
            if isinstance(f, (list, tuple)):
                f = f[0]
            feats.append(f.detach().cpu())
            pids.extend(batch_pids.numpy().tolist())
            camids.extend(batch_camids.numpy().tolist())
    feats = torch.cat(feats, dim=0).numpy()
    return feats, np.array(pids), np.array(camids)


def main():
    t0 = time.time()
    base = Path(DATASET_BASE)
    query_dir = base / "query"
    gallery_dir = base / "bounding_box_test"
    if not query_dir.exists() or not gallery_dir.exists():
        print(f"[ERROR] Could not find {query_dir} or {gallery_dir}. Set MARKET1501_DIR env var to the dataset root.")
        sys.exit(1)

    transform = T.Compose([
        T.Resize((256, 128)) if False else T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    q_imgs = _list_images(str(query_dir))
    g_imgs = _list_images(str(gallery_dir))

    qset = ImgDataset(q_imgs, transform)
    gset = ImgDataset(g_imgs, transform)

    qloader = DataLoader(qset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    gloader = DataLoader(gset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    net = FeatureExtractor()
    if MODEL_WEIGHTS and os.path.isfile(MODEL_WEIGHTS):
        ckpt = torch.load(MODEL_WEIGHTS, map_location="cpu")
        missing, unexpected = net.load_state_dict(ckpt, strict=False)
        print(f"[INFO] Loaded weights: missing={len(missing)} unexpected={len(unexpected)}")
    net.to(DEVICE)

    print("[INFO] Extracting query features...")
    qf, q_pids, q_camids = extract_features(net, qloader)

    print("[INFO] Extracting gallery features...")
    gf, g_pids, g_camids = extract_features(net, gloader)

    print("[INFO] Evaluating...")
    metrics = evaluate(qf, q_pids, q_camids, gf, g_pids, g_camids, topk=(1,5,10,20,50,100))

    dt = time.time() - t0
    print("==== Market-1501 Results ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"[Done] Elapsed: {dt:.1f}s")

if __name__ == "__main__":
    import numpy as np
    main()
