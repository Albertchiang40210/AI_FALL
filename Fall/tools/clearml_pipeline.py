import os
import sys

# =========================================================================
# 🛑 核心防線：禁止 ClearML 翻 Git，並開啟非同步上傳
# =========================================================================
os.environ["CLEARML_GIT_BYPASS"] = "1"
os.environ["CLEARML_LOG_MODEL_ASYNC"] = "1"

from clearml import Task
from ultralytics import RTDETR

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pretrain_weights = os.path.join(BASE_DIR, "rtdetr-l.pt")
    
    if not os.path.exists(pretrain_weights):
        print(f"❌ 錯誤：找不到官方預訓練權重 {pretrain_weights}，請確認檔案在專案目錄下！")
        sys.exit(1)

    # =========================================================================
    # 📊 1. 初始化 ClearML 實驗任務
    # =========================================================================
    print("🔄 [ClearML] 正在初始化地端實驗追蹤任務...")
    task = Task.init(
        project_name="Nursing_Home_Fall_Detection", 
        task_name="RT-DETR_Official_Weights_Validation"
    )

    # =========================================================================
    # 🧠 2. 載入官方大腦
    # =========================================================================
    print("\n🏗️  [模型建構] 正在直接從 .pt 權重還原網路架構...")
    model = RTDETR(pretrain_weights)

    # =========================================================================
    # 🚀 3. 核心變更：改用 val() 驗證模式，徹底繞過地端無 label 的報錯
    # =========================================================================
    print("\n⚡ [管線測試] 正在利用官方內建 coco8 數據集衝刺 ClearML 數據綁定...")
    
    # 🎯 這裡吃 coco8.yaml，它只有 8 張圖，下載極快，且自帶 txt 標籤，完全不依賴你電腦裡的路徑
    metrics = model.val(
        data="coco8.yaml", 
        imgsz=640,
        device="cpu", # 測試用 CPU 最穩定防死鎖
        project="Nursing_Home_Fall_Detection",
        name="clearml_val_test"
    )

    print("\n🎉 [全線通車] 官方模型驗證完成！")
    print("📦 所有精確度指標（mAP50, mAP50-95）已全自動同步至 ClearML 雲端控制台。")
    
    task.close()

if __name__ == "__main__":
    main()