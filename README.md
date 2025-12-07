# AI-CUP 2025 – YOLOv8 五摺交叉驗證訓練與加權框融合 (WBF) 管線
# Group 9580

本專案實作了完整的 **5-Fold Cross-Validation (交叉驗證)** 訓練與推論流程，使用 **YOLOv8** 模型進行物件偵測，並透過 **Weighted Box Fusion (WBF)** 將五個摺訓練結果融合為最終輸出。

---

## 專案大致結構

```
AI-CUP-2025_9580/
├─ training.py              # 進行5-Fold CrossValidation訓練的主程式
├─ testing.py               # 使用各最佳權重對測試集進行預測
├─ WBF_submissions.py       # 將五個預測結果進行加權框融合 (WBF)
├─ AICUP_2025_9580/
│   ├─ YOLOv8s_5Fold_CV/    # 儲存5-Fold的訓練結果(訓練過後)
│   ├─ submission_1.txt     # 第1摺的預測結果(預測過後)
│   ├─ submission_2.txt     # 第2摺的預測結果(預測過後)
│   ├─ submission_3.txt     # 第3摺的預測結果(預測過後)
│   ├─ submission_4.txt     # 第4摺的預測結果(預測過後)
│   ├─ submission_5.txt     # 第5摺的預測結果(預測過後)
│   └─ submission_final_fused.txt  # 經WBF融合後的最終輸出
```

---

## 整體流程說明

### 1. 5-Fold CrossValidation模型訓練 (`training.py`)
此程式以病人為單位，進行5-Fold交叉驗證，且確保不同Fold的病人資料互不重疊，盡可能的讓模型學習各種不同的情況。

- 重複五次，每Fold都會自動生成：
  - 訓練 / 驗證的 `.txt` 名單
  - 專屬的 `fold_k_data.yaml` 設定檔
  - YOLOv8 訓練過程及最佳權重 (`best.pt`)

輸出結果會儲存於：
```
AICUP_2025_9580/YOLOv8s_5Fold_CV/fold_{k}/weights/best.pt
```

---

### 2. 測試集預測 (`testing.py`)
完成訓練後，依序使用每個摺的最佳模型權重對測試集進行推論。

在程式開頭依序修改路徑，執行五次：
```python
WEIGHTS = "AICUP_2025_9580/YOLOv8s_5Fold_CV/fold_1/weights/best.pt"
OUT_TXT = "AICUP_2025_9580/submission_1.txt"
```
分別執行 `fold_1` 到 `fold_5`。

輸出結果會儲存於：
```
AICUP_2025_9580/submission_1.txt(到submission_5.txt)
```

---

### 3. 加權框融合策略 (WBF) (`WBF_submissions.py`)
完成對測試集的預測後，使用WBF(Weight Box Fusion)將五個預測結果加權融合。

- 融合的輸入：
  ```
  submission_1.txt ~ submission_5.txt
  ```
- 根據各摺的 `mAP50` 與 `mAP50-95` 進行加權。
- 生成最終輸出：
  ```
  AICUP_2025_9580/submission_final_fused.txt
  ```

**主要設定：**
- IoU門檻：`0.5`
- Skip Box門檻：`0.2`
- 加權方式：以摺別性能指標(mAP)作為加權依據(手動輸入)
- 可選設定：
  - `KEEP_TOP1_PER_IMG` → 每張影像只保留置信度最高的1個框(嘗試後推薦不使用，過度限制了模型預測(容易遺漏positive)

---

## 環境需求

**安裝依賴：**
```bash
pip install ultralytics ensemble-boxes tqdm numpy
```

**Python版本：** `>=3.8`  
**硬體建議：** 支援CUDA的GPU以加速YOLOv8訓練與推論。

---

## 輸出範例

### 各Fold輸出 (`submission_1.txt`)
```
000001 0 0.943217 112 98 412 412
000002 0 0.856793 225 133 382 379
```

### 最終融合結果 (`submission_final_fused.txt`)
```
000001 0 0.963521 115 100 410 409
000002 0 0.887462 228 134 380 376
```

---

## 備註

- 加權方式計算如下：
  ```
  weight = α * mAP50 + (1 - α) * mAP50-95
  ```
  其中 α 預設為 0.7，可隨情況調整。
- 權重在融合前會自動正規化。
- 若希望僅保留Top-1框，可將 `KEEP_TOP1_PER_IMG` 設為 `True`。

---

作者：李云揚 、 賴瑾蓉 、 蔡宜臻  
更新日期：2025/12/07
