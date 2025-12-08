"""
File: training.py
Created on 2025-12-07
Description:
    This script performs 5-fold cross-validation training for YOLOv8.
    using patient-level splits for AI-CUP 2025.
@Author: 賴瑾蓉 、 李云揚 、 蔡宜臻
"""
import os
from pathlib import Path
import json
import numpy as np
from ultralytics import YOLO

# --- 參數設定 ---
K_FOLDS = 5
PATIENT_MAP_FILE = Path("AICUP_2025_9580/new_dataset/patient_map.json")
LABELS_DIR = Path("AICUP_2025_9580/new_dataset/labels")
SPLITS_DIR = Path("AICUP_2025_9580/kfold_splits") # 儲存 .txt 和 .yaml 的地方
PROJECT_NAME = "YOLOv8s_5Fold_CV"
# --------------------

# --- YOLO訓練參數 ---
TRAIN_PARAMS = {
    "model": "yolov8s.pt", #選擇使用yolov8s
    "device": 1,
    "workers": 8,
    "batch": 32,
    "imgsz": 640,
    "epochs": 100,
    "patience": 50,
    "save_period": 0,
    "augment": True,
    "mosaic": 1.0,
    "mixup": 0.1,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "dropout": 0.1,
    "lr0": 0.01,         
    "lrf": 0.01,        
    "weight_decay": 0.0005
}

# ---------------------------------

def main():
    print("Loading patient map...")
    with open(PATIENT_MAP_FILE, 'r') as f:
        patient_map = json.load(f)
    
    # 取得所有 50 位病患 ID 並打亂
    patient_ids = sorted(patient_map.keys())
    np.random.seed(42) # 確保每次隨機都一樣
    np.random.shuffle(patient_ids)
    
    # 將 50 位病患分成 5 Folds
    folds = np.array_split(patient_ids, K_FOLDS)
    
    print(f"Created {K_FOLDS} folds with {len(folds[0])} patients each.")
    
    # 建立儲存 split 檔案的資料夾
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_fold_results = []

    # --- 執行 5-Fold CV 迴圈 ---
    for k in range(K_FOLDS):
        print("\n" + "="*50)
        print(f" STARTING FOLD {k+1}/{K_FOLDS} ")
        print("="*50)

        # 1. 決定 Train/Val 病患
        val_patients = folds[k]
        train_patients = np.concatenate([folds[i] for i in range(K_FOLDS) if i != k])
        
        print(f"Val patients: {val_patients}")
        print(f"Train patients: {len(train_patients)} total")

        # 2. 生成 .txt 檔案以供讀取與核對每個Fold的病患資料分布 
        train_txt_path = SPLITS_DIR / f"fold_{k+1}_train.txt"
        val_txt_path = SPLITS_DIR / f"fold_{k+1}_val.txt"
        
        with open(train_txt_path, 'w') as f_train:
            for patient_id in train_patients:
                for img_path in patient_map[patient_id]:
                    f_train.write(f"{img_path}\n")

        with open(val_txt_path, 'w') as f_val:
            for patient_id in val_patients:
                for img_path in patient_map[patient_id]:
                    f_val.write(f"{img_path}\n")
        
        print(f"Wrote {len(train_patients)} patients to {train_txt_path}")
        print(f"Wrote {len(val_patients)} patients to {val_txt_path}")

        # 3. 生成 data.yaml 檔案
        yaml_path = SPLITS_DIR / f"fold_{k+1}_data.yaml"
        with open(yaml_path, 'w') as f_yaml:
            f_yaml.write(f"train: {train_txt_path.resolve()}\n")
            f_yaml.write(f"val: {val_txt_path.resolve()}\n")
            f_yaml.write(f"nc: 1\n")
            f_yaml.write(f"names: ['aortic_valve']\n")
            
 
            # 取得Label資訊。
            f_yaml.write(f"label_dir: {LABELS_DIR.resolve()}\n")

        # 4. 載入模型並開始訓練
        model = YOLO(TRAIN_PARAMS['model'])
        
        # 傳入所有其他參數
        train_args = TRAIN_PARAMS.copy()
        train_args['data'] = str(yaml_path)
        train_args['project'] = PROJECT_NAME
        train_args['name'] = f"fold_{k+1}"
        
        del train_args['model'] 

        results = model.train(**train_args)

if __name__ == "__main__":
    main()
