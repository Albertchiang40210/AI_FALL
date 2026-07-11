import os
import sys
from pathlib import Path
import cv2
import requests

# 🚨 核心大一統：將原本的 SAM 2 汰換，全面換裝為與前線和 ClearML 後台重訓同構的 DEIM-DETR (RT-DETR)
from ultralytics import RTDETR

# =========================================================================
# 1. 參數與環境變數配置區
# =========================================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

def load_dotenv(path: Path) -> None:
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

load_dotenv(PROJECT_ROOT / ".env")

LS_URL = os.getenv("LS_URL", "http://localhost:8085") 
PROJECT_ID = int(os.getenv("LS_PROJECT_ID", "1").strip())  
CONF_THRES = float(os.getenv("CONF_THRES", "0.35")) # 方形框環境偵測建議 0.35

# 🎯 【Label Studio 網頁登入帳密】
USERNAME = "wang4021096@gmail.com"  
PASSWORD = "Topenglish86021"     

# 🚨 核心更換：預設權重改為與前線完全對齊的 rtdetr-l.pt
MODEL_PATH = str(PROJECT_ROOT / "rtdetr-l.pt")

IMAGES_DIR = PROJECT_ROOT / "active_learning_dataset" / "images"
LABELS_DIR = PROJECT_ROOT / "active_learning_dataset" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# 📊 擴充環境與長輩偵測所需的 COCO 類別映射字典 (只篩選病房關鍵物件，徹底過濾雜訊)
ENVIRONMENT_COCO = {
    0: "person",
    56: "chair",
    57: "sofa",
    59: "bed",
    62: "tv"
}

# 💾 對應地端 YOLO/RT-DETR 方形框標準重新訓練的類別 ID 映射
ENV_LABEL_TO_YOLO = {
    "person": 0,
    "chair": 1,
    "sofa": 2,
    "bed": 3,
    "tv": 4
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def fail(msg: str) -> None:
    print(f"\n[X] {msg}")
    sys.exit(1)

# =========================================================================
# 2. 模擬瀏覽器登入
# =========================================================================
print(f"[*] 正在建立 Session 並嘗試登入 {LS_URL} ...")
session = requests.Session()

login_page_url = f"{LS_URL}/user/login/"
try:
    init_res = session.get(login_page_url, timeout=5)
    csrftoken = session.cookies.get('csrftoken', '')
except Exception as e:
    fail(f"無法連線至 Label Studio 服務: {e}")

login_data = {
    "email": USERNAME,
    "password": PASSWORD,
    "csrfmiddlewaretoken": csrftoken
}
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": login_page_url
})

login_res = session.post(login_page_url, data=login_data, allow_redirects=True)

if "login" in login_res.url:
    fail("帳號或密碼錯誤，請檢查 USERNAME 和 PASSWORD 設定！")

print("🎉 [登入成功] 已獲取合法網頁 Session 憑證！")

# =========================================================================
# 3. 撈取與匯入圖片
# =========================================================================
tasks_url = f"{LS_URL}/api/projects/{PROJECT_ID}/tasks/"
tasks_res = session.get(tasks_url, params={"page_size": 1000}, timeout=5)
existing_tasks = tasks_res.json()
if isinstance(existing_tasks, dict) and "results" in existing_tasks:
    existing_tasks = existing_tasks["results"]

imported_names = {Path(t["data"].get("image", "")).name for t in existing_tasks}

if not IMAGES_DIR.exists():
    fail(f"找不到影像資料夾: {IMAGES_DIR}")

local_images = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
if not local_images:
    fail(f"{IMAGES_DIR} 沒有任何圖片")

new_tasks = [
    {"image": f"/data/local-files/?d=images/{p.name}"}
    for p in local_images
    if p.name not in imported_names
]

if new_tasks:
    import_url = f"{LS_URL}/api/projects/{PROJECT_ID}/import/"
    session.headers.update({"X-CSRFToken": session.cookies.get('csrftoken', '')})
    session.post(import_url, json=new_tasks, timeout=5)
    print(f"[+] 匯入 {len(new_tasks)} 張新圖片為 task")
    
    tasks_res = session.get(tasks_url, params={"page_size": 1000}, timeout=5)
    existing_tasks = tasks_res.json()
    if isinstance(existing_tasks, dict) and "results" in existing_tasks:
        existing_tasks = existing_tasks["results"]
else:
    print("[=] 沒有新圖片需要匯入")

# =========================================================================
# 4. 對全專案 task 跑 DEIM-DETR 推論 (與 ClearML 後台重訓完全同構)
# =========================================================================
pending = existing_tasks 
print(f"[*] 共 {len(pending)} 個 task 將使用 DEIM-DETR 進行環境智慧方形框標註")

if not pending:
    print("[=] 沒有任務需要處理，結束。")
    sys.exit(0)

print(f"[*] 載入 DEIM-DETR 智慧環境偵測大腦： {MODEL_PATH} ...")
model = RTDETR(MODEL_PATH)
model_version = f"deim-rtdetr-{Path(MODEL_PATH).stem}-env-bbox"

pushed = 0
for idx, task in enumerate(pending, 1):
    filename = Path(task["data"].get("image", "")).name
    img_path = IMAGES_DIR / filename
    if not img_path.exists(): continue

    img = cv2.imread(str(img_path))
    if img is None: continue
    img_h, img_w, _ = img.shape

    # 執行 DEIM-DETR 物件偵測
    results = model.predict(str(img_path), conf=CONF_THRES, verbose=False)

    ls_result = []
    yolo_lines = []
    
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            # 🚨 只篩選出長照核心環境物件，其餘天花板、地板、雜物直接過濾，徹底排除雜訊！
            if cls_id not in ENVIRONMENT_COCO: continue
            
            label_name = ENVIRONMENT_COCO[cls_id]
            conf = float(box.conf[0].item())
            
            # 獲取方形框座標 xyxy
            xyxy = box.xyxy.cpu().numpy()[0]
            # 🚨 關鍵修正 1：第一時間將 NumPy float32 強制轉為標準 Python 原生 float
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            
            # 🚨 關鍵修正 2：確保所有百分比計算與 round() 出來的結果均為標準 float
            ls_x = float(round((x1 / img_w) * 100, 4))
            ls_y = float(round((y1 / img_h) * 100, 4))
            ls_w = float(round(((x2 - x1) / img_w) * 100, 4))
            ls_h = float(round(((y2 - y1) / img_h) * 100, 4))
            
            # 塞入正宗的 rectanglelabels JSON 結構
            ls_result.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",  # 🚨 核心大統：從多邊形換回最穩定的方形框！
                "value": {
                    "x": ls_x,
                    "y": ls_y,
                    "width": ls_w,
                    "height": ls_h,
                    "rectanglelabels": [label_name]
                },
                "score": conf
            })
            
            # 2. 轉換為地端 YOLO/RT-DETR 物件偵測重訓標準格式
            yolo_id = ENV_LABEL_TO_YOLO[label_name]
            xywh_norm = box.xywhn.cpu().numpy()[0]
            cx, cy, bw, bh = float(xywh_norm[0]), float(xywh_norm[1]), float(xywh_norm[2]), float(xywh_norm[3])
            yolo_lines.append(f"{yolo_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    # 💾 寫回地端標準 YOLO .txt 檔案 (這群標籤隨後可以直接餵進 ClearML 重新 fine-tune 訓練)
    (LABELS_DIR / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

    # 🌐 灌回網頁
    pred_payload = {
        "project": PROJECT_ID,
        "task": task["id"],
        "result": ls_result,
        "model_version": model_version
    }
    
    pred_url = f"{LS_URL}/api/predictions/"
    session.headers.update({"X-CSRFToken": session.cookies.get('csrftoken', '')})
    res_pred = session.post(pred_url, json=pred_payload, timeout=5)
    
    if res_pred.status_code in [200, 201]:
        pushed += 1
        print(f"  [{idx}/{len(pending)}] {filename} → 智慧框選出 {len(ls_result)} 個關鍵長照物件 (DEIM-DETR 注入成功)")
    else:
        print(f"  [!] {filename} 網頁注入失敗，狀態碼: {res_pred.status_code}")

print(f"\n[OK] 大功告成！共更新 {pushed} 個專案的 DEIM-DETR 預標註。請回到網頁並重整檢視成果！")