import os
from pathlib import Path
from dotenv import load_dotenv
from clearml import Dataset, Task

# =====================================================================
# 1. 精準路徑防呆：鎖定最外層 FALL/
# =====================================================================
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent  # 從 tools/ 往上推兩層回到 FALL/

# 載入 .env
if (PROJECT_ROOT / ".env").exists():
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# 🚨 正式鎖定：外層的黃金資料夾
DATASET_DIR = PROJECT_ROOT / "active_learning_dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

print(f"[INFO] 專案根目錄定位成功: {PROJECT_ROOT}")
print(f"[INFO] 正在讀取外層資料集: {DATASET_DIR}")

# 嚴格檢查外層資料夾是否存在
if not DATASET_DIR.exists():
    print(f"❌ 錯誤：找不到外層的 active_learning_dataset！請確認路徑。")
    exit(1)

# =====================================================================
# 2. ClearML 資料集上傳與版本化
# =====================================================================
print("\n--- 步驟一：正在將外層最新標籤上傳至 ClearML ---")

try:
    cl_dataset = Dataset.create(
        dataset_project="Active_Learning_Fall_Detection",
        dataset_name="YOLO_Fall_Dataset",
        description="透過 Label Studio 人工審核確認後的最新主動學習訓練集"
    )
    
    # 這裡的 local_base_folder 設為 DATASET_DIR，會保持內部 images/ 與 labels/ 的結構
    cl_dataset.add_files(path=str(IMAGES_DIR), local_base_folder=str(DATASET_DIR))
    cl_dataset.add_files(path=str(LABELS_DIR), local_base_folder=str(DATASET_DIR))
    
    print("[INFO] 正在上傳變更至 ClearML 儲存端...")
    cl_dataset.upload()
    cl_dataset.finalize()
    
    dataset_id = cl_dataset.id
    print(f"🎉 ClearML 資料集建立成功！最新 ID: {dataset_id}")

except Exception as e:
    print(f"❌ ClearML 資料集處理失敗: {e}")
    exit(1)

# =====================================================================
# 3. 觸發 ClearML RT-DETR 自動重訓
# =====================================================================
print("\n--- 步驟二：正在觸發 ClearML RT-DETR 自動重訓任務 ---")

try:
    training_task = Task.get_task(
        project_name="Active_Learning_Fall_Detection",
        task_name="RT-DETR_DEIM_Training_Base"
    )
    
    cloned_task = Task.clone(source_task=training_task, name=f"RT-DETR_Retrain_Dataset_{dataset_id[:8]}")
    
    hyper_params = cloned_task.get_parameters()
    hyper_params["Args/dataset_id"] = dataset_id  
    cloned_task.set_parameters(hyper_params)
    
    Task.enqueue(cloned_task, queue_name="default")
    
    print(f"🚀 自動重訓任務已成功推入 ClearML 佇列！")
    print(f"🔗 任務連結: {cloned_task.get_output_log_web_page()}")

except Exception as e:
    print(f"❌ 觸發重訓失敗: {e}")
    print("[💡 提示] 步驟一已成功！若步驟二報錯，代表你還沒在 ClearML 後台建立名為 'RT-DETR_DEIM_Training_Base' 的任務範本，這不影響資料已成功版本化的事實。")