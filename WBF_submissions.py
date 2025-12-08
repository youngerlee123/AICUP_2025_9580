"""
File: WBF_submissions.py
Created on 2025-12-07
Description:
    This script performs weighted boxes fusion for 5-fold testing prediction with YOLOv8.
    using patient-level splits for AI-CUP 2025.
@Author: 賴瑾蓉 、 李云揚 、 蔡宜臻
"""
import os
from pathlib import Path
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm

SUBMISSION_FILES = [
    "AICUP_2025_9580/submission_1.txt", #Fold1
    "AICUP_2025_9580/submission_2.txt", #Fold2
    "AICUP_2025_9580/submission_3.txt", #Fold3
    "AICUP_2025_9580/submission_4.txt", #Fold4
    "AICUP_2025_9580/submission_5.txt", #Fold5
]

FINAL_SUBMISSION = "AICUP_2025_9580/submission_final_fused.txt"

IMG_WIDTH  = 512.0
IMG_HEIGHT = 512.0

WBF_IOU_THR       = 0.5
WBF_SKIP_BOX_THR  = 0.2
KEEP_TOP1_PER_IMG = False     # << 每張圖最多保留 1 個框 關掉目前看來比較好
MIN_VOTES         = 1        # << 至少多少模型對同一物件投票；若不想限制設為 1

#Weights，根據5-Fold"實際的訓練結果"將Best.pt的Val mAP50與mAP50-95填入
fold_metrics = np.array([[0.967,0.677],
                         [0.947,0.602],
                         [0.959,0.631],
                         [0.963,0.664],
                         [0.918,0.550]])
alpha = 0.7
weights = alpha*fold_metrics[:,0] + (1-alpha)*fold_metrics[:,1]
WEIGHTS = weights / weights.sum()

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

def main():
    print("Starting 5-Fold Ensemble using WBF...")
    image_predictions = {}
    K_FOLDS = len(SUBMISSION_FILES)

    print(f"Reading {K_FOLDS} submission files...")
    per_model_counts = [0] * K_FOLDS

    for file_index, file_path in enumerate(SUBMISSION_FILES):
        p = Path(file_path)
        if not p.exists():
            print(f"Warning: not found {p}, skip.")
            continue

        with p.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 7:
                    continue
                img_stem = parts[0]
                label    = int(parts[1])
                score    = float(parts[2])
                x1, y1, x2, y2 = map(float, parts[3:7])

                if img_stem not in image_predictions:
                    image_predictions[img_stem] = {
                        "boxes_lists":  [[] for _ in range(K_FOLDS)],
                        "scores_lists": [[] for _ in range(K_FOLDS)],
                        "labels_lists": [[] for _ in range(K_FOLDS)],
                    }

                box_norm = [x1/IMG_WIDTH, y1/IMG_HEIGHT, x2/IMG_WIDTH, y2/IMG_HEIGHT]
                image_predictions[img_stem]["boxes_lists"][file_index].append(box_norm)
                image_predictions[img_stem]["scores_lists"][file_index].append(score)
                image_predictions[img_stem]["labels_lists"][file_index].append(label)
                per_model_counts[file_index] += 1

    print(f"Found predictions for {len(image_predictions)} images.")
    print("Per-model raw box counts:", per_model_counts)

    if WEIGHTS is not None:
        w = np.array(WEIGHTS, dtype=float)
        WEIGHTS[:] = list((w / (w.sum() + 1e-9)).tolist())

    total_written = 0
    with open(FINAL_SUBMISSION, "w", encoding="utf-8", newline="\n") as f_out:
        for img_stem, data in tqdm(image_predictions.items(), desc="Fusing"):
            boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
                data["boxes_lists"],
                data["scores_lists"],
                data["labels_lists"],
                weights=WEIGHTS,           # None 或正規化後的權重
                iou_thr=WBF_IOU_THR,
                skip_box_thr=WBF_SKIP_BOX_THR,
                conf_type="avg"            
            )

            if len(scores_fused) == 0:
                continue

            # --- 過濾最少投票數（對每個 fused box 檢查有幾個模型支持）--- 目前實測不做這個(也就是調為1)是最好的。
            if MIN_VOTES > 1:
                kept_b, kept_s, kept_l = [], [], []
                # 把各模型原始框還原到像素座標，供投票數計算
                orig_boxes_per_model = []
                for m in range(K_FOLDS):
                    orig_boxes = []
                    for bn in data["boxes_lists"][m]:
                        orig_boxes.append([bn[0]*IMG_WIDTH, bn[1]*IMG_HEIGHT,
                                           bn[2]*IMG_WIDTH, bn[3]*IMG_HEIGHT])
                    orig_boxes_per_model.append(orig_boxes)

                for b, s, l in zip(boxes_fused, scores_fused, labels_fused):
                    fused_px = [b[0]*IMG_WIDTH, b[1]*IMG_HEIGHT, b[2]*IMG_WIDTH, b[3]*IMG_HEIGHT]
                    votes = 0
                    for m in range(K_FOLDS):
                        if any(iou_xyxy(fused_px, ob) >= WBF_IOU_THR for ob in orig_boxes_per_model[m]):
                            votes += 1
                    if votes >= MIN_VOTES:
                        kept_b.append(b); kept_s.append(s); kept_l.append(l)
                boxes_fused, scores_fused, labels_fused = kept_b, kept_s, kept_l

            # --- 每張圖只留 Top-1 目前為False不使用---
            if KEEP_TOP1_PER_IMG and len(scores_fused) > 1:
                top = int(np.argmax(scores_fused))
                boxes_fused  = [boxes_fused[top]]
                scores_fused = [scores_fused[top]]
                labels_fused = [labels_fused[top]]



            # --- 寫檔 ---
            for box, score, label in zip(boxes_fused, scores_fused, labels_fused):
                x1 = int(round(box[0] * IMG_WIDTH))
                y1 = int(round(box[1] * IMG_HEIGHT))
                x2 = int(round(box[2] * IMG_WIDTH))
                y2 = int(round(box[3] * IMG_HEIGHT))
                x1 = max(0, min(x1, int(IMG_WIDTH-1)))
                y1 = max(0, min(y1, int(IMG_HEIGHT-1)))
                x2 = max(0, min(x2, int(IMG_WIDTH-1)))
                y2 = max(0, min(y2, int(IMG_HEIGHT-1)))
                f_out.write(f"{img_stem} {int(label)} {float(score):.6f} {x1} {y1} {x2} {y2}\n")
                total_written += 1

    print("\n" + "="*50)
    print(f"WBF done. Total boxes written: {total_written}")
    print(f"saved: {FINAL_SUBMISSION}")

if __name__ == "__main__":
    main()
