from clearml import Task
from ultralytics import RTDETR
import torch

def main():
    # 1. 檢查硬體加速環境
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️  Detected device: {device}")

    # 2. 初始化 ClearML 任務
    # 建立任務名稱，方便在 ClearML UI 中進行對照組比對
    task = Task.init(
        project_name="Nursing_Home_Project", 
        task_name="RT-DETR_DEIM_M5Pro_Optimized"
    )
    
    print("🚀 [ClearML] 任務初始化完成，準備進入訓練...")

    # 3. 載入模型 (RT-DETR Large)
    model = RTDETR("rtdetr-l.pt")
    
    # 4. 啟動訓練
    # 針對 M5 Pro 48G 記憶體進行的參數微調：
    # - batch=16: 充分利用 48G 統一記憶體
    # - half=True: 已更換為新版相容性寫法，底層會自動啟用最優硬體加速
    # - workers=8: 根據 M5 Pro 的多核心優勢進行數據預處理
    model.train(
        data="tools/data.yaml", # 🎯 已精準導向至 tools 內的設定檔
        epochs=50,             # 根據實際需求調整
        imgsz=640,
        batch=16,              # 若記憶體溢出 (OOM) 可調回 8
        device=device,
        workers=8,
        
        # 效能與收斂優化
        augment=True,          # 啟動增強策略
        val=True,              # 開啟訓練中評估
        plots=True,            # 自動產生績效圖表
        
        # 訓練紀錄優化
        cache=True             # 將數據載入記憶體，減少磁碟讀寫延遲
    )
    
    print("🎉 [Success] 訓練流程結束，模型權重已同步至 ClearML。")

if __name__ == "__main__":
    main()