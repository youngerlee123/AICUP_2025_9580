"""
File: training.py
Created on 2025-12-07
Description:
    This script consolidates all training images and labels into a unified new dataset.
    Builds a patient-level image mapping for subsequent 5-fold cross-validation training.
@Author: 賴瑾蓉 、 李云揚 、 蔡宜臻
"""
import os
import shutil
from pathlib import Path
import json
from tqdm import tqdm

# 原始影像路徑
IMAGES_ROOT = Path("AICUP_2025_9580/training_image")
LABELS_ROOT = Path("AICUP_2025_9580/training_label")

# 複製到新的路徑方便後續訓練
MASTER_IMAGES_DIR = Path("AICUP_2025_9580/new_dataset/images")
MASTER_LABELS_DIR = Path("AICUP_2025_9580/new_dataset/labels")
PATIENT_MAP_FILE = Path("AICUP_2025_9580/new_dataset/patient_map.json")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# --------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_label_for_image(img_path: Path, labels_root: Path, patient_id: str) -> Path | None:
    base = img_path.stem
    cand1 = labels_root / patient_id / f"{base}.txt"
    if cand1.exists():
        return cand1
    cand2 = labels_root / f"{base}.txt"
    if cand2.exists():
        return cand2
    return None

def main():
    ensure_dir(MASTER_IMAGES_DIR)
    ensure_dir(MASTER_LABELS_DIR)

    patients = sorted([d for d in IMAGES_ROOT.iterdir() if d.is_dir() and d.name.startswith('patient')])
    print(f"Found {len(patients)} patients. Processing...")

    patient_map = {} # 提供每個病患影像的路徑讓後續訓練可以讀取
    
    for patient_dir in tqdm(patients, desc="Processing patients"):
        patient_id = patient_dir.name
        patient_map[patient_id] = []

        for img in sorted(patient_dir.rglob("*")):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                continue

            # 為了避免重名相關的錯誤，使用 patientID___filename.png
            parts = img.relative_to(patient_dir).parts
            new_name_base = patient_id + "___" + "__".join(parts)
            new_img_name = new_name_base
            new_lbl_name = Path(new_name_base).stem + ".txt"

            # 1. 複製影像
            dst_img_path = MASTER_IMAGES_DIR / new_img_name
            shutil.copy2(str(img), str(dst_img_path))
            patient_map[patient_id].append(str(dst_img_path))

            # 2. 檢查並複製Label (只有正樣本有)
            label_src = find_label_for_image(img, LABELS_ROOT, patient_id)
            if label_src:
                dst_lbl_path = MASTER_LABELS_DIR / new_lbl_name
                shutil.copy2(str(label_src), str(dst_lbl_path))

    # 3. 儲存Patient map
    with open(PATIENT_MAP_FILE, 'w') as f:
        json.dump(patient_map, f, indent=2)

    print("\n Dataset setup complete.")
    print(f"All images copied to: {MASTER_IMAGES_DIR}")
    print(f"All labels copied to: {MASTER_LABELS_DIR}")
    print(f"Patient map saved to: {PATIENT_MAP_FILE}")

if __name__ == "__main__":
    main()