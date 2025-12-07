"""
File: testing.py
Created on 2025-12-07
Description:
    This script performs 5-fold testing prediction with YOLOv8.
    using patient-level splits for AI-CUP 2025.
@Author: 李云揚 、 賴瑾蓉 、 蔡宜臻
"""
import os
from pathlib import Path
import torch
from ultralytics import YOLO
import numpy as np

#使用Cross validation訓練完的YOLOv8s_5Fold_CV裡面找到Fold1-Fold5的最佳權重，依序對Testing set進行預測
WEIGHTS = "AICUP_2025_9580/YOLOv8s_5Fold_CV/fold_1/weights/best.pt" #fold_1-5
OUT_TXT = "AICUP_2025_9580/submission_1.txt" #1-5

def main():
    model = YOLO(WEIGHTS)

    results_gen = model.predict(
        source="testing_image/**/*", # Testing set
        imgsz=640, 
        conf=0.25,
        device=0,
        save=False,
        stream=True,
        verbose=False
    )

    out_path = Path(OUT_TXT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_imgs, n_boxes = 0, 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in results_gen:
            n_imgs += 1
            img_stem = Path(r.path).stem

            if r.boxes is None or len(r.boxes) == 0:
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), score in zip(xyxy, confs):
                # ---------- Clip 座標至影像邊界 ----------
                h, w = r.orig_shape  # (height, width)
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # 轉為整數
                x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

                cls_id = 0
                n_boxes += 1
                f.write(f"{img_stem} {cls_id} {float(score):.6f} {x1i} {y1i} {x2i} {y2i}\n")

    print(f"Done. Images processed: {n_imgs}, boxes written: {n_boxes}")
    print(f"Submission saved to: {OUT_TXT}")

if __name__ == "__main__":
    main()
