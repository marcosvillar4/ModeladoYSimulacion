from django.urls import path

from .views import SimulationListView, SimulationResultView, SimulationRunView

app_name = "simulations"

urlpatterns = [
    path("", SimulationListView.as_view(), name="list"),
    path("<slug:slug>/", SimulationRunView.as_view(), name="run"),
    path("<slug:slug>/resultado/<uuid:run_id>/", SimulationResultView.as_view(), name="result"),
]

