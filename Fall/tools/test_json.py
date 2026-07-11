import json

def generate_db_injection_script():
    prediction_result = [
        {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": 1920,
            "original_height": 1080,
            "value": {
                "x": 38.0,
                "y": 50.0,
                "width": 20.0,
                "height": 35.0,
                "rectanglelabels": ["person"]
            }
        }
    ]
    
    serialized_result = json.dumps(prediction_result)
    
    # 💡 鎖定單張測試：直接用 Task.objects.get(id=4)
    django_code = f"""def run_injection():
    from django.apps import apps
    from django.db import transaction
    Prediction = apps.get_model('tasks', 'Prediction')
    Task = apps.get_model('tasks', 'Task')
    
    try:
        # 🎯 精準狙擊：直接抓你網頁上看到的 ID 4
        target_task = Task.objects.get(id=4)
        
        with transaction.atomic():
            Prediction.objects.filter(task=target_task).delete()
            Prediction.objects.create(
                task=target_task, 
                project=target_task.project, 
                model_version='RT-DETR-DEIM-V1', 
                score=0.85, 
                result={serialized_result}
            )
        print("\\n==== 🎉 SUCCESS: 成功注入 Task ID", target_task.id, "====\\n")
    except Task.DoesNotExist:
        print("\\n==== ❌ ERROR: 找不到 ID 為 4 的任務 ====\\n")
    except Exception as e:
        print("\\n==== ❌ ERROR: 未知錯誤", e, "====\\n")

run_injection()
"""

    with open("inject.py", "w", encoding="utf-8") as f:
        f.write(django_code)
        
    print("💾 成功生成【單張 ID: 4 狙擊版】密令：inject.py")

if __name__ == "__main__":
    generate_db_injection_script()