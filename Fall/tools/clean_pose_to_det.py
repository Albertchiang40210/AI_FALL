# clean_pose_to_det.py
import os
import shutil

src_dir = "./active_learning_dataset"
dst_dir = "./detection_dataset"

# 1. 複製整個結構（包含圖片）
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
shutil.copytree(src_dir, dst_dir)

print("📁 資料夾複製完成，開始剝離 Pose 關鍵點...")

# 2. 清洗 labels 底下的所有 .txt
labels_path = os.path.join(dst_dir, "labels")
if os.path.exists(labels_path):
    for filename in os.listdir(labels_path):
        if filename.endswith(".txt"):
            file_key = os.path.join(labels_path, filename)
            
            with open(file_key, "r") as f:
                lines = f.readlines()
            
            clean_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 關鍵動作：只切前 5 個欄位 (class, x, y, w, h)
                    clean_lines.append(" ".join(parts[:5]))
            
            with open(file_key, "w") as f:
                f.write("\n".join(clean_lines))

print("🎯 降維清洗完成！已生成純偵測資料集於: ./detection_dataset")