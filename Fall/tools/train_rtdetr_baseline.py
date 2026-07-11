# train_rtdetr_baseline.py
import os
from clearml import Task
from ultralytics import RTDETR

def main():
    # 1. 初始化 ClearML 任務，自動追蹤接下來的所有指標與權重
    task = Task.init(
        project_name="Nursing_Home_Project", 
        task_name="RT-DETR_Baseline_Run"
    )
    
    print("🚀 [ClearML] 任務初始化成功，開始載入 RT-DETR 模型...")

    # 2. 載入 RT-DETR 官方預訓練權重（大腦換裝：從 YOLO 轉 Transformer）
    # 第一次執行會自動下載 rtdetr-l.pt 到目前目錄
    model = RTDETR("rtdetr-l.pt")
    
    # 3. 開始訓練（直接吃你原本 YOLO 的資料集配置檔案）
    # 請將 'your_dataset.yaml' 替換成你實際的 YOLO 格式資料集路徑（例如 data.yaml）
    # 先設定 epochs=1 用來通電測試，確認 M 晶片與 ClearML 沒噴錯再開大
    dataset_yaml = "data.yaml" 
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ 找不到資料集設定檔: {dataset_yaml}，請修正路徑！")
        return

    print("🏋️ [Train] 開始進行 1 Epoch 的環境通電測試...")
    model.train(
        data=dataset_yaml,
        epochs=1,        # 通電測試用，成功後再改成 100
        imgsz=640,
        batch=4,         # 顧慮到 Mac 本地記憶體，先從 4 或 8 開始
        device="cpu"     # Mac M 晶片在 Ultralytics 中跑 RT-DETR 建議先用 cpu 或 mps 測試
    )
    
    print("🎉 [Success] RT-DETR Baseline 通電成功！請檢查 ClearML 看板。")

if __name__ == "__main__":
    main()