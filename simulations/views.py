import uuid

from django.contrib import messages
from django.http import Http404
from django.shortcuts import redirect, render
from django.views import View
from django.views.generic import TemplateView
from plotly.offline import get_plotlyjs

from .forms import DynamicSimulationForm
from .services import (
    SimulationInputError,
    execute_simulation,
    get_simulation,
    list_simulations,
    register_default_simulations,
)


_SESSION_RUNS_KEY = "simulation_runs"


class SimulationListView(TemplateView):
    template_name = "simulations/list.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        register_default_simulations()
        context["simulations"] = list_simulations()
        return context


class SimulationRunView(View):
    template_name = "simulations/run.html"

    @staticmethod
    def _expression_names(spec) -> list[str]:
        return [expression.name for expression in spec.expressions]

    @staticmethod
    def _get_session_runs(request) -> dict:
        return request.session.get(_SESSION_RUNS_KEY, {})

    @staticmethod
    def _save_payload_to_session(request, run_id: str, payload: dict) -> None:
        runs = SimulationRunView._get_session_runs(request)
        runs[run_id] = payload
        request.session[_SESSION_RUNS_KEY] = runs

    def _render_form(self, request, spec, form):
        return render(
            request,
            self.template_name,
            {
                "spec": spec,
                "form": form,
                "expression_names": self._expression_names(spec),
            },
        )

    def get(self, request, slug: str):
        spec = self._get_spec(slug)
        form = DynamicSimulationForm(spec=spec)
        return self._render_form(request, spec, form)

    def post(self, request, slug: str):
        spec = self._get_spec(slug)
        form = DynamicSimulationForm(spec=spec, data=request.POST)

        if not form.is_valid():
            return self._render_form(request, spec, form)

        try:
            payload = execute_simulation(slug, form.cleaned_data)
        except SimulationInputError as exc:
            form.add_error(None, str(exc))
            return self._render_form(request, spec, form)

        run_id = str(uuid.uuid4())
        self._save_payload_to_session(request, run_id, payload)
        messages.success(request, "Simulacion ejecutada correctamente.")
        return redirect("simulations:result", slug=slug, run_id=run_id)

    @staticmethod
    def _get_spec(slug: str):
        register_default_simulations()
        try:
            return get_simulation(slug)
        except KeyError as exc:
            raise Http404("Simulacion no encontrada") from exc


class SimulationResultView(TemplateView):
    template_name = "simulations/result.html"

    @staticmethod
    def _resolve_expression_names(payload: dict) -> list[str]:
        register_default_simulations()
        try:
            spec = get_simulation(payload["slug"])
        except KeyError:
            return []
        return [expression.name for expression in spec.expressions]

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        run_id = str(self.kwargs["run_id"])
        runs = self.request.session.get(_SESSION_RUNS_KEY, {})
        payload = runs.get(run_id)

        if payload is None:
            raise Http404("No existe resultado para la ejecucion solicitada")

        context["expression_names"] = self._resolve_expression_names(payload)
        context["payload"] = payload
        context["plotly_js"] = get_plotlyjs()
        return context
