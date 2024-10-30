import torch
import os
from django.conf import settings
from .models.lstm_model import LSTMModel





# 设置模型路径
model_path = os.path.join(settings.BASE_DIR, 'MyCCF/training/models/LSTMModel.pth')

# 实例化模型并加载权重
model = LSTMModel(9,64,2,1)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))  # 如果在CPU上运行
model.eval()  # 设置为评估模式



from django.http import JsonResponse
import torch

def predict_training(request):
    if request.method == 'POST':
        # 从表单数据中获取输入值并转化为张量
        features = [
            float(request.POST['steps']),
            float(request.POST['exercise_time']),
            float(request.POST['avg_heart_rate']),
            float(request.POST['max_heart_rate']),
            float(request.POST['sleep_duration']),
            float(request.POST['fatigue_level']),
            float(request.POST['height']),
            float(request.POST['weight']),
            float(request.POST['age'])
        ]
        # input_tensor = torch.tensor([features])
        input_tensor = torch.tensor([[features]], dtype=torch.float32)

        # 模型预测
        with torch.no_grad():
            prediction = model(input_tensor)
            print("Model prediction:", prediction.item())  # 打印预测结果以便调试
            result = int(prediction.item() > 0.5)  # 假设输出大于 0.22 表示需要训练

        return JsonResponse({'result': result})

    return render(request, 'training/predict.html')

from django.shortcuts import render

def home(request):
    return render(request, 'training/home.html')
