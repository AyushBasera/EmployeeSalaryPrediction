from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_salary, name='predict'),
    path('eda/',views.eda_dashboard_view,name="eda"),
]
