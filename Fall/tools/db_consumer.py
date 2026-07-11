# db_consumer.py
import json
import signal
import sys
from kafka import KafkaConsumer
from tools.db_manager import DatabaseManager

print("🚀 [MLOps 數據落地引擎] 正在初始化服務...")

# 1. 初始化資料庫管理員並檢查/建立資料表
db = DatabaseManager()
if not db.init_table():
    print("❌ 資料庫初始化失敗，程式強行中斷。")
    sys.exit(1)

# 2. 啟動 Kafka Consumer，專職監聽所有最終報告
consumer = KafkaConsumer(
    'processed-reports',  # 👈 盯緊邊緣快速道路與 VLM 二審發出的最終 Topic
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='latest',
    group_id='db-storage-cluster'  # 儲存專用的群組 ID
)

print("📥 異步落地監聽器已上線！正在監控 Kafka 管道，準備將 AI 數據寫入 PostgreSQL...")

# 優雅關閉處理機制
def signal_handler(sig, frame):
    print("\n🛑 收到終止訊號，正在安全關閉落地監聽器...")
    consumer.close()
    print("👋 服務已完全停止。")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 3. 數據消費無窮循環
try:
    for message in consumer:
        final_report = message.value
        print(f"\n📥 [Kafka 2 頻道] 攔截到全新 AI 通報 (Device: {final_report.get('device_id')})")
        
        # 👈 完美對接：調用你的 db_manager.py 進行解構與非同步落地
        db.log_report(final_report)
        
except Exception as e:
    print(f"❌ 數據落地監聽循環異常中斷: {e}")