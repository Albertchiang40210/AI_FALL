import json
import time
import os
import shutil
import ollama  
from kafka import KafkaConsumer, KafkaProducer
from clearml import Task  
from ultralytics import RTDETR  # 👈 換裝：導入全新 Transformer 核心引擎

# =========================================================================
# 🎛️ 1. MLOps 核心初始化（工業級監控追蹤）
# =========================================================================
task = Task.init(
    project_name="Nursing_Home_Project", 
    task_name="VLM_Brain_Worker_Scheme_B",
    tags=["Kafka", "Qwen2.5-VL", "Active-Learning", "RT-DETR", "Scheme_B"] 
)

print("📦 [原生 Video-LLM + RT-DETR 方案 B 核心啟動] 正在連線地端 Kafka 流數據引擎...")

# 2. 監聽前段 Kafka 1
consumer = KafkaConsumer(
    'nursing-home-alerts',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda v: json.loads(v.decode('utf-8')), 
    auto_offset_reset='latest',  
    group_id='vlm-brain-cluster'
)

# 3. 準備轉發到後段 Kafka 2
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 💡 核心大腦雙引擎宣告對齊
VLM_MODEL_NAME = "qwen2.5-vl"
DETECTOR_MODEL = RTDETR("rtdetr-l.pt") 

print(f"🚀 [護理長大腦上線] 監聽中... 推論引擎為 RT-DETR，多模態為 {VLM_MODEL_NAME}！")


# =========================================================================
# 🧠 核心功能：主動學習打包引擎（Active Learning Engine）- 方案 B JSON 預測框版
# =========================================================================
def package_active_learning_sample(img_path, camera_id, rtdetr_box_data):
    """
    方案 B：維持 images/ 與 labels/ 拆開。
    直接在 active_learning_dataset 內建立一個 predictions/ 資料夾，
    並寫入 Label Studio 原生認得的 Pre-annotation JSON 檔案。
    """
    try:
        dataset_base = "active_learning_dataset"
        os.makedirs(f"{dataset_base}/images", exist_ok=True)
        os.makedirs(f"{dataset_base}/predictions", exist_ok=True)
        
        base_name = os.path.basename(img_path)
        json_name = base_name + ".json"  # 範例：snapshot_101.jpg.json
        
        # 1. 複製實體相片到 images 夾
        shutil.copy(img_path, f"{dataset_base}/images/{base_name}")
        
        # 2. 如果沒有偵測到框，就不建立預測 JSON
        if not rtdetr_box_data:
            return

        # 💡 解析 RT-DETR 傳進來的 xywhn 座標與類別
        cls_id = int(rtdetr_box_data.cls[0])
        xywhn = rtdetr_box_data.xywhn[0].tolist()
        
        # 關鍵轉換：YOLO 的 x_center, y_center 轉為 Label Studio 的左上角 x, y
        x_max_100 = (xywhn[0] - (xywhn[2] / 2)) * 100
        y_max_100 = (xywhn[1] - (xywhn[3] / 2)) * 100
        w_max_100 = xywhn[2] * 100
        h_max_100 = xywhn[3] * 100
        
        # 類別 ID 對應名稱對齊
        label_map = {0: "person", 1: "bed", 2: "wheelchair"}
        label_name = label_map.get(cls_id, "person")

        # 3. 封裝成 Label Studio 官方標準的預測 Schema
        label_studio_json = {
            "result": [
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x_max_100,
                        "y": y_max_100,
                        "width": w_max_100,
                        "height": h_max_100,
                        "rectanglelabels": [label_name]
                    }
                }
            ],
            "score": float(rtdetr_box_data.conf[0]) # 帶入 RT-DETR 信心度
        }
        
        # 4. 寫入 JSON 檔案
        with open(f"{dataset_base}/predictions/{json_name}", "w") as f:
            json.dump(label_studio_json, f, indent=2)
            
        print(f"💾 [方案 B 閉環] 成功打包預測 JSON 檔案：{json_name}")
    except Exception as e:
        print(f"⚠️ [方案 B 閉環] 打包預測 JSON 失敗: {e}")


# =========================================================================
# 📥 主數據流監聽循環
# =========================================================================
for message in consumer:
    event_data = message.value
    
    alert_type = event_data.get("event_type", "Pending_VLM_Review") 
    cam_id = event_data.get("camera_id", "Unknown_Room")           
    env_clues = event_data.get("event_type", "No specific objects") 
    
    try:
        clean_device_id = int(cam_id) if str(cam_id).isdigit() else 101
    except:
        clean_device_id = 101

    # 🎯 取得流數據路徑與截圖路徑
    clip_path = event_data.get("clip_path", "/vids/fallback.mp4")
    image_filename = event_data.get("image_filename") or event_data.get("snapshot_path")
    base_dir = "/Users/albert/Documents/專案/AIPE03/Fall"
    
    # 🎯 終極修復點：智慧相容判斷。如果前端給的就是絕對路徑，直接使用！
    if image_filename and os.path.isabs(image_filename):
        img_path = image_filename
    elif image_filename:
        img_path = os.path.join(base_dir, image_filename)
    else:
        img_path = os.path.join(base_dir, f"snapshot_{cam_id}.jpg")

    full_clip_path = os.path.join(base_dir, clip_path.lstrip("/")) if not os.path.isabs(clip_path) else clip_path

    if not os.path.exists(img_path):
        print(f"⚠️ 找不到邊緣截圖：{img_path}，跳過此幀。")
        continue

    print(f"📸 [VLM 大腦] 成功定點命中相片，開始進行 RT-DETR 推理: {img_path}")

    # RT-DETR 推論
    det_results = DETECTOR_MODEL(img_path, imgsz=640, verbose=False)[0]
    boxes = det_results.boxes

    highest_score = 0.0
    best_box = None
    
    if len(boxes) > 0:
        highest_score = float(boxes.conf.max())
        best_box_idx = boxes.conf.argmax()
        best_box = boxes[best_box_idx]  # 💡 直接抓取最高分數的 Box 物件

    confidence_pct = f"{highest_score * 100:.1f}%"

    # 🚦 邊緣端信心度三區間分流策略
    RTDETR_HIGH_CONFIDENCE = 0.75  
    RTDETR_LOW_CONFIDENCE = 0.35   

    raw_report = None
    severity = "low"
    should_send_report = True

    # 🟢 狀況一：定時巡檢環境
    if alert_type == "Routine_Environment_Sanity_Check":
        print(f"\n[🔍 定時環境巡檢] 房間：{cam_id}。")
        prompt_text = (
            "You must reply ONLY in Traditional Chinese (繁體中文).\n"
            "You are an AI head nurse conducting a routine security check. Inspect if there are any potential environmental hazards.\n"
            "Please output a structured environment report using this exact template:\n\n"
            "【安養中心智慧環境巡檢報告】\n"
            f"1. 巡檢相機: {cam_id}\n"
            "2. 巡檢狀態: 正常 / 發現潛在隱患\n"
            "3. 現場環境具體描述: \n"
            "4. 預防性護理建議: "
        )
        
        try:
            response = ollama.chat(
                model=VLM_MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt_text, 'images': [img_path]}]
            )
            raw_report = response['message']['content'].strip()
        except Exception as e:
            raw_report = f"【系統警告】巡檢推理中斷。原因: {str(e)}"

    # 🔴 狀況二：屬於動態警報事件（如跌倒）
    else:
        # A 區間：高置信度直接過閘外發
        if highest_score >= RTDETR_HIGH_CONFIDENCE:
            print(f"\n[🟢 RT-DETR 高置信度直接通報] 房間：{cam_id}。分數：{confidence_pct}，跳過 VLM。")
            raw_report = f"【AI 快速通報】RT-DETR 高置信度解耦偵測 ({confidence_pct}) 觸發 {alert_type} 事件。系統判定風險極高，已跳過 Video-LLM 複核，秒級推播警報！"
            severity = "high" if "fall" in alert_type.lower() else "low"

        # B 區間：模糊樣本區間，觸發 Qwen2.5-VL 原生影片時序二審 + 方案 B 閉環
        elif RTDETR_LOW_CONFIDENCE <= highest_score < RTDETR_HIGH_CONFIDENCE:
            print(f"\n[🟡 觸發 RT-DETR 模糊二審區間] 房間：{cam_id}。分數：{confidence_pct}，啟動原生 Video-LLM 影片審查...")
            
            if not os.path.exists(full_clip_path):
                print(f"⚠️ 找不到實體影片：{full_clip_path}，降級使用 Snapshot。")
                vlm_input_source = [img_path]
            else:
                vlm_input_source = [full_clip_path]

            prompt_text = (
                "You must reply ONLY in Traditional Chinese (繁體中文).\n"
                "You are an AI head nurse in a security care center. "
                "Please watch this security video clip carefully to evaluate the patient's dynamic safety.\n"
                "Analyze the behavioral changes over time (e.g., whether the person actually fell, slipped slowly, or just bent over).\n"
                f"Edge system clues: {env_clues}.\n\n"
                "Output a structured alert report using this exact template:\n\n"
                "【安養中心緊急通報（Video-LLM 原生影片二審版）】\n"
                f"1. 通報相機: {cam_id}\n"
                "2. 影片動態行為確認 (請詳細描述整個動作的因果與发生過程): \n"
                "3. 現場風險評級 (例如: 緊急 / 潛在風險 / 誤報攔截): \n"
                f"4. AI 時序判讀信心度: (邊緣置信度為 {confidence_pct}，請重新評估)\n"
                "5. 醫療建議行動: "
            )
            
            try:
                start_time = time.time()
                response = ollama.chat(
                    model=VLM_MODEL_NAME,
                    messages=[{'role': 'user', 'content': prompt_text, 'images': vlm_input_source}]
                )
                raw_report = response['message']['content'].strip()
                print(f"✨ [{cam_id}] 原生 Video-LLM 二審完成，耗時 {time.time() - start_time:.2f} 秒。")
                
                severity = "high" if "high" in raw_report.lower() or "緊急" in raw_report or "跌倒" in raw_report else "low"
                
                # 🚀 數據閉環：呼叫新版方案 B 函數，直接傳入 best_box 物件
                if best_box is not None:
                    package_active_learning_sample(img_path, cam_id, best_box)
                        
            except Exception as e:
                print(f"❌ Ollama 原生 Video-LLM 推理失敗: {str(e)}")
                raw_report = f"【系統容錯提示】Video-LLM 服務異常。房間: {cam_id}，依據邊緣端預警強制發報。原因: {str(e)}"
                severity = "high"

        # C 區間：分數過低，判定為純雜訊直接攔截
        else:
            print(f"\n[🔴 雜訊攔截] 房間：{cam_id}。RT-DETR 置信度過低 ({confidence_pct})，已成功過濾。")
            should_send_report = False

    # =========================================================================
    # 📢 外發 Kafka 2 管道
    # =========================================================================
    if should_send_report and (raw_report is not None):
        iso_detected_at = event_data.get("detected_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

        final_report = {
            "device_id": clean_device_id,                                
            "event_type": alert_type,                                    
            "clip_path": clip_path,  
            "detected_at": iso_detected_at,                              
            "snapshot_path": img_path,                                   
            "yolo_score": highest_score,                                
            "yolo_threshold": RTDETR_LOW_CONFIDENCE,  
            "vlm_summary": raw_report,  
            "severity": severity         
        }
        
        producer.send('processed-reports', value=final_report)
        producer.flush()
        print(f"📢 [Kafka 2] 雙軌審查報告已外發！已完成 EventCreateRequest 工業級規格轉換！")