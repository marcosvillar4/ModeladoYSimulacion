from django.urls import path

from .views import AboutView, HomeView

app_name = "core"

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("acerca-de/", AboutView.as_view(), name="about"),
]

