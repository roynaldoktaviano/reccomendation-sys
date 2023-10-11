from django.urls import path
from . import views

urlpatterns = [
    path('', views.indexs, name='indexs'),
    path('diagnose', views.diagnose, name='diagnose'),
]
