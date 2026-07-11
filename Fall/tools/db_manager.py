# db_manager.py
import psycopg2
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        # 💡 未來組員提供正式的環境資訊時，你只需要修改這 5 個變數：
        self.host = "localhost"
        self.database = "aidb"
        self.user = "testuser"
        self.password = "testpassword"
        self.port = "5433"

    def _get_connection(self):
        """內部連線工廠：建立並回傳資料庫連線"""
        return psycopg2.connect(
            host=self.host, 
            database=self.database,
            user=self.user, 
            password=self.password, 
            port=self.port
        )

    def init_table(self) -> bool:
        """初始化資料表，確保測試環境完整"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 配合你的 final_report 規格建立完美的欄位
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ai_processed_reports (
                        id SERIAL PRIMARY KEY,
                        device_id INT,
                        event_type VARCHAR(100),
                        yolo_score FLOAT,
                        vlm_summary TEXT,
                        snapshot_path VARCHAR(255),
                        detected_at TIMESTAMP,
                        created_at TIMESTAMP
                    );
                    """)
                    conn.commit()
            print("💾 [PostgreSQL] 資料表 ai_processed_reports 初始化/檢查成功！")
            return True
        except Exception as e:
            print(f"❌ [DB Init] 資料表外掛初始化失敗: {e}")
            return False

    def log_report(self, report_data: dict):
        """核心寫入方法：直接解構你的 final_report 字典，好維護、不搞硬編碼"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                    INSERT INTO ai_processed_reports 
                    (device_id, event_type, yolo_score, vlm_summary, snapshot_path, detected_at, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """
                    cursor.execute(query, (
                        report_data.get("device_id"),
                        report_data.get("event_type"),
                        report_data.get("yolo_score"),
                        report_data.get("vlm_summary"),
                        report_data.get("snapshot_path"),
                        report_data.get("detected_at"),  # 傳入 ISO 字串，PGSQL 會自動轉為 Timestamp
                        datetime.now()                     # 本地落地儲存時間
                    ))
                    conn.commit()
            print(f"💾 [PostgreSQL] 雙軌審查數據已成功落地儲存！(Device: {report_data.get('device_id')})")
        except Exception as e:
            print(f"❌ [PostgreSQL] 獨立資料模組寫入失敗: {e}")