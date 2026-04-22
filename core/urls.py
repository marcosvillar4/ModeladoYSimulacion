from django.urls import path

from .views import AboutView, HomeView, MathJaxReferenceView

app_name = "core"

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("referencia-mathjax/", MathJaxReferenceView.as_view(), name="mathjax_reference"),
    path("acerca-de/", AboutView.as_view(), name="about"),
]

