"""
Real-time Pothole Detection using Gradient-based Edge Mapping
CS305 - Computer Vision (IFHE Hyderabad)

Pipeline
  1. Resize + ROI crop (road region, lower part of the frame)
  2. Grayscale + CLAHE (illumination normalisation)
  3. Gaussian blur (noise suppression before differentiation)
  4. Gradient computation with Sobel operators (Gx, Gy -> magnitude, direction)
  5. Edge map: gradient magnitude threshold + Canny (hysteresis, non-max suppression)
  6. Morphological closing + hole filling to turn broken edge rings into regions
  7. Contour extraction and geometric / texture filtering
  8. Output: bounding boxes, overlay, per-frame latency and FPS

Usage
  python pothole_detector.py --image road.jpg
  python pothole_detector.py --video road.mp4 --save out.mp4
  python pothole_detector.py --camera 0
  python pothole_detector.py --folder images/ --gt labels/ --out results/   (batch + IoU eval)
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

# ----------------------------- parameters ---------------------------------
PARAMS = dict(
    width=640,            # working width (keeps real-time speed)
    roi_top=0.45,         # ignore top 45 % of frame (sky / horizon)
    blur_ksize=5,
    sobel_ksize=3,
    canny_low=50,
    canny_high=140,
    close_ksize=9,
    min_area_frac=0.002,  # min blob area as fraction of ROI area
    max_area_frac=0.35,   # reject huge blobs (shadows, whole-road regions)
    min_solidity=0.55,    # area / convex-hull area
    min_extent=0.30,      # area / bounding-box area
    max_aspect=4.0,       # w/h or h/w limit
    min_edge_density=0.06 # fraction of edge pixels along the blob boundary band
)


# ----------------------------- core steps ---------------------------------
def preprocess(frame, p=PARAMS):
    h0, w0 = frame.shape[:2]
    scale = p["width"] / float(w0)
    frame = cv2.resize(frame, (p["width"], int(h0 * scale)))
    y0 = int(frame.shape[0] * p["roi_top"])
    roi = frame[y0:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (p["blur_ksize"],) * 2, 0)
    return frame, roi, gray, y0


def gradient_edge_map(gray, p=PARAMS):
    """Sobel gradients -> magnitude / direction, then combined edge map."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=p["sobel_ksize"])
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=p["sobel_ksize"])
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    mag8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Otsu on magnitude gives a data-driven threshold for strong gradients
    _, strong = cv2.threshold(mag8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny = cv2.Canny(gray, p["canny_low"], p["canny_high"])
    edges = cv2.bitwise_and(strong, canny) if strong.any() else canny
    edges = cv2.bitwise_or(edges, canny)  # keep Canny's thin, connected edges
    return mag8, ang, edges


def edges_to_regions(edges, p=PARAMS):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["close_ksize"],) * 2)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    # fill closed rings -> candidate pothole regions
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, k)
    return closed, filled


def filter_candidates(filled, edges, gray, p=PARAMS):
    """Geometric + edge-density filtering of candidate contours."""
    H, W = filled.shape
    roi_area = float(H * W)
    dets = []
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if not (p["min_area_frac"] * roi_area <= area <= p["max_area_frac"] * roi_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w / float(h), h / float(w))
        extent = area / float(w * h)
        hull = cv2.convexHull(c)
        solidity = area / max(cv2.contourArea(hull), 1.0)
        # edge density in a thin band around the boundary
        band = np.zeros_like(filled)
        cv2.drawContours(band, [c], -1, 255, thickness=5)
        density = cv2.countNonZero(cv2.bitwise_and(edges, band)) / max(cv2.countNonZero(band), 1)
        # interior texture: potholes are usually darker / rougher than surrounding asphalt
        m = np.zeros_like(filled)
        cv2.drawContours(m, [c], -1, 255, cv2.FILLED)
        inside = gray[m > 0]
        ring = cv2.dilate(m, np.ones((15, 15), np.uint8)) - m
        outside = gray[ring > 0]
        darker = (inside.mean() < outside.mean()) if outside.size else True
        if (aspect <= p["max_aspect"] and solidity >= p["min_solidity"]
                and extent >= p["min_extent"] and density >= p["min_edge_density"] and darker):
            dets.append(dict(box=(x, y, w, h), area=area, solidity=solidity,
                             density=density, contour=c))
    return dets


def detect(frame, p=PARAMS):
    t0 = time.perf_counter()
    frame_r, roi, gray, y0 = preprocess(frame, p)
    mag, ang, edges = gradient_edge_map(gray, p)
    closed, filled = edges_to_regions(edges, p)
    dets = filter_candidates(filled, edges, gray, p)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    stages = dict(gray=gray, magnitude=mag, edges=edges, closed=closed, filled=filled)
    return frame_r, y0, dets, stages, latency_ms


def draw(frame_r, y0, dets, latency_ms):
    out = frame_r.copy()
    for d in dets:
        x, y, w, h = d["box"]
        cv2.rectangle(out, (x, y + y0), (x + w, y + y0 + h), (0, 0, 255), 2)
        cv2.putText(out, "pothole", (x, max(y + y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0
    cv2.putText(out, f"{latency_ms:.1f} ms | {fps:.0f} FPS | {len(dets)} det",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# ----------------------------- evaluation ---------------------------------
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def load_yolo_labels(path, W, H):
    """YOLO txt: class cx cy w h (normalised). Returns list of (x, y, w, h) in pixels."""
    boxes = []
    if os.path.exists(path):
        for line in open(path):
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            boxes.append((int((cx - w / 2) * W), int((cy - h / 2) * H), int(w * W), int(h * H)))
    return boxes


def evaluate_folder(folder, gt_folder, out_dir, iou_thr=0.3):
    os.makedirs(out_dir, exist_ok=True)
    TP = FP = FN = 0
    lat = []
    files = sorted(sum([glob.glob(os.path.join(folder, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []))
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        frame_r, y0, dets, _, ms = detect(img)
        lat.append(ms)
        H, W = frame_r.shape[:2]
        gt = load_yolo_labels(os.path.join(gt_folder, os.path.splitext(os.path.basename(f))[0] + ".txt"), W, H)
        pred = [(x, y + y0, w, h) for (x, y, w, h) in (d["box"] for d in dets)]
        matched = set()
        for pb in pred:
            best, bj = 0, -1
            for j, g in enumerate(gt):
                if j in matched:
                    continue
                v = iou(pb, g)
                if v > best:
                    best, bj = v, j
            if best >= iou_thr:
                TP += 1
                matched.add(bj)
            else:
                FP += 1
        FN += len(gt) - len(matched)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), draw(frame_r, y0, dets, ms))
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"Images: {len(files)}  TP={TP} FP={FP} FN={FN}")
    print(f"Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")
    if lat:
        print(f"Latency mean={np.mean(lat):.1f} ms  median={np.median(lat):.1f} ms  "
              f"-> {1000 / np.mean(lat):.1f} FPS")


# ----------------------------- CLI ----------------------------------------
def run_stream(src, save=None):
    cap = cv2.VideoCapture(src)
    writer = None
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_r, y0, dets, _, ms = detect(frame)
        vis = draw(frame_r, y0, dets, ms)
        if save and writer is None:
            writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), 25,
                                     (vis.shape[1], vis.shape[0]))
        if writer:
            writer.write(vis)
        cv2.imshow("Pothole detection (q to quit)", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--folder")
    ap.add_argument("--gt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--save")
    a = ap.parse_args()
    if a.image:
        img = cv2.imread(a.image)
        fr, y0, dets, st, ms = detect(img)
        os.makedirs(a.out, exist_ok=True)
        cv2.imwrite(os.path.join(a.out, "detected.jpg"), draw(fr, y0, dets, ms))
        for k, v in st.items():
            cv2.imwrite(os.path.join(a.out, f"stage_{k}.png"), v)
        print(f"{len(dets)} detections, {ms:.1f} ms")
    elif a.video:
        run_stream(a.video, a.save)
    elif a.camera is not None:
        run_stream(a.camera, a.save)
    elif a.folder and a.gt:
        evaluate_folder(a.folder, a.gt, a.out)
    else:
        ap.print_help()
