from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_training, name='predict_training'),
    path('', views.home, name='home'),  # 添加默认视图
]
