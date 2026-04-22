from django.test import TestCase
from django.urls import reverse
from unittest.mock import patch

from .services import register_default_simulations


class SimulationsViewsTest(TestCase):
    def setUp(self):
        register_default_simulations()

    def test_list_page_renders(self):
        response = self.client.get(reverse("simulations:list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Simulaciones disponibles")
        self.assertContains(response, "Metodo de punto fijo")
        self.assertContains(response, "Integracion de Monte Carlo doble")

    def test_run_page_shows_live_math_preview(self):
        response = self.client.get(reverse("simulations:run", kwargs={"slug": "punto-fijo"}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-math-input="true"')
        self.assertContains(response, 'data-math-preview-for="id_gx"')
        self.assertContains(response, "vendor/mathjax/tex-svg.js")
        self.assertContains(response, "formula_preview.js")

    def test_seed_field_only_for_stochastic_simulations(self):
        deterministic_response = self.client.get(reverse("simulations:run", kwargs={"slug": "euler"}))
        self.assertEqual(deterministic_response.status_code, 200)
        self.assertNotContains(deterministic_response, 'name="seed"')

        stochastic_response = self.client.get(reverse("simulations:run", kwargs={"slug": "monte-carlo-simple"}))
        self.assertEqual(stochastic_response.status_code, 200)
        self.assertContains(stochastic_response, 'name="seed"')

    def test_monte_carlo_methods_show_confidence_level_input(self):
        response_simple = self.client.get(reverse("simulations:run", kwargs={"slug": "monte-carlo-simple"}))
        self.assertEqual(response_simple.status_code, 200)
        self.assertContains(response_simple, 'name="confidence_level"')

        response_double = self.client.get(reverse("simulations:run", kwargs={"slug": "monte-carlo-doble"}))
        self.assertEqual(response_double.status_code, 200)
        self.assertContains(response_double, 'name="confidence_level"')

    def test_integration_methods_show_xi_input(self):
        response = self.client.get(reverse("simulations:run", kwargs={"slug": "trapecio-simple"}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="xi"')

    def test_run_fixed_point_redirects_to_result(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "punto-fijo"}),
            data={
                "gx": "cos(x)",
                "x0": 0.5,
                "a": -0.5,
                "b": 1.5,
                "tol": 1e-6,
                "max_iter": 50,
                "seed": 42,
                "precision": 6,
            },
        )
        self.assertEqual(response.status_code, 302)
        redirect_url = response["Location"]
        self.assertIn("resultado", redirect_url)

        result_response = self.client.get(redirect_url)
        self.assertEqual(result_response.status_code, 200)
        self.assertContains(result_response, "Resultado: Metodo de punto fijo")

    def test_run_fixed_point_accepts_latex_like_input(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "punto-fijo"}),
            data={
                "gx": r"\frac{1}{2}x",
                "x0": 2.0,
                "a": 0.0,
                "b": 2.5,
                "tol": 1e-6,
                "max_iter": 50,
                "seed": 42,
                "precision": 6,
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("resultado", response["Location"])

    def test_run_bisection_with_invalid_symbol_shows_error(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "biseccion"}),
            data={"fx": "sin(z)", "a": 1, "b": 2, "tol": 1e-6, "max_iter": 20, "seed": 42, "precision": 6},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No se pudo interpretar la expresion")

    def test_run_bisection_renders_function_and_convergence_plot(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "biseccion"}),
            data={"fx": "x**3 - x - 2", "a": 1, "b": 2, "tol": 1e-6, "max_iter": 20, "seed": 42, "precision": 6},
        )
        self.assertEqual(response.status_code, 302)

        runs = self.client.session.get("simulation_runs", {})
        payload = runs[next(iter(runs))]
        self.assertIn("plotly-graph-div", payload["result"]["plot"])
        self.assertIn("Error por iteracion en biseccion", payload["result"]["plot"])
        self.assertIn("|f(c_n)|", payload["result"]["plot"])

        aux_plots = payload["result"].get("auxiliary_plots", [])
        self.assertEqual(len(aux_plots), 1)
        self.assertEqual(aux_plots[0]["title"], "Funcion original f(x)")
        self.assertIn("plotly-graph-div", aux_plots[0]["plot"])

        result_response = self.client.get(response["Location"])
        self.assertEqual(result_response.status_code, 200)
        self.assertContains(result_response, "Resultado: Metodo de biseccion")
        self.assertContains(result_response, "Muestra de resultados")

    def test_bisection_plot_uses_original_interval_bounds(self):
        with patch("simulations.services._aux_plot_original_function", return_value="<div>mock plot</div>") as mocked_plot:
            response = self.client.post(
                reverse("simulations:run", kwargs={"slug": "biseccion"}),
                data={"fx": "x**3 - x - 2", "a": 1, "b": 2, "tol": 1e-6, "max_iter": 20, "seed": 42, "precision": 6},
            )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(mocked_plot.called)
        _, plot_a, plot_b, plot_title = mocked_plot.call_args.args[:4]
        self.assertEqual(plot_a, 1.0)
        self.assertEqual(plot_b, 2.0)
        self.assertEqual(plot_title, "Funcion original en el intervalo [a, b]")

    def test_run_euler_redirects_to_result(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "euler"}),
            data={"fxy": "x+y", "x0": 0.0, "y0": 1.0, "h": 0.1, "n": 10, "seed": 42, "precision": 6},
        )
        self.assertEqual(response.status_code, 302)
        redirect_url = response["Location"]
        self.assertIn("resultado", redirect_url)

        runs = self.client.session.get("simulation_runs", {})
        payload = runs[next(iter(runs))]
        self.assertEqual(
            payload["result"]["table"]["headers"],
            ["n", "x_n", "y_n", "y_n+1", "f(x_n,y_n)", "error estimado"],
        )

        result_response = self.client.get(redirect_url)
        self.assertEqual(result_response.status_code, 200)
        self.assertContains(result_response, "Resultado: Metodo de Euler")
        self.assertContains(result_response, "error estimado")
        self.assertContains(result_response, "\\(x_n\\)")
        self.assertContains(result_response, "plotly-graph-div")

    def test_run_rk4_with_invalid_expression_shows_error(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "runge-kutta-4"}),
            data={"fxy": "x+z", "x0": 0.0, "y0": 1.0, "h": 0.1, "n": 10, "seed": 42, "precision": 6},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No se pudo interpretar la expresion")

    def test_run_heun_redirects_to_result(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "heun-euler-mejorado"}),
            data={"fxy": "x+y", "x0": 0.0, "y0": 1.0, "h": 0.1, "n": 10, "seed": 42, "precision": 6},
        )
        self.assertEqual(response.status_code, 302)

        result_response = self.client.get(response["Location"])
        self.assertEqual(result_response.status_code, 200)
        self.assertContains(result_response, "Resultado: Metodo de Heun")

    def test_precision_parameter_rounds_numeric_results(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "euler"}),
            data={"fxy": "x+y", "x0": 0.0, "y0": 1.0, "h": 0.1, "n": 10, "seed": 42, "precision": 2},
        )
        self.assertEqual(response.status_code, 302)

        runs = self.client.session.get("simulation_runs", {})
        payload = runs[next(iter(runs))]
        final_y = payload["result"]["summary"]["y aproximado final"]
        self.assertEqual(final_y, round(final_y, 2))

    def test_list_page_uses_configured_order(self):
        response = self.client.get(reverse("simulations:list"))
        self.assertEqual(response.status_code, 200)

        slugs = [spec.slug for spec in response.context["simulations"]]
        self.assertEqual(
            slugs,
            [
                "punto-fijo",
                "biseccion",
                "newton-raphson",
                "aitken-delta-cuadrado",
                "interpolacion-lagrange",
                "newton-diferencias-divididas",
                "trapecio-simple",
                "trapecio-compuesta",
                "simpson-13-simple",
                "simpson-13-compuesta",
                "simpson-38-simple",
                "simpson-38-compuesta",
                "regla-rectangulo",
                "monte-carlo-simple",
                "monte-carlo-doble",
                "euler",
                "heun-euler-mejorado",
                "runge-kutta-4",
            ],
        )

    def test_integration_methods_render_in_single_chart(self):
        runs_to_check = [
            (
                "trapecio-compuesta",
                {
                    "fx": "sin(x)",
                    "a": 0.0,
                    "b": 3.14159,
                    "n": 6,
                    "xi": 1.5707963267948966,
                    "seed": 42,
                    "precision": 6,
                },
            ),
            (
                "simpson-13-compuesta",
                {
                    "fx": "sin(x)",
                    "a": 0.0,
                    "b": 3.14159,
                    "n": 6,
                    "xi": 1.5707963267948966,
                    "seed": 42,
                    "precision": 6,
                },
            ),
            (
                "regla-rectangulo",
                {
                    "fx": "sin(x)",
                    "a": 0.0,
                    "b": 3.14159,
                    "n": 8,
                    "xi": 1.5707963267948966,
                    "seed": 42,
                    "precision": 6,
                },
            ),
        ]

        for slug, payload in runs_to_check:
            response = self.client.post(reverse("simulations:run", kwargs={"slug": slug}), data=payload)
            self.assertEqual(response.status_code, 302)

            runs = self.client.session.get("simulation_runs", {})
            self.assertTrue(runs)
            stored_payload = runs[next(reversed(runs))]
            aux_plots = stored_payload["result"].get("auxiliary_plots", [])
            self.assertEqual(aux_plots, [])
            self.assertIn("plotly-graph-div", str(stored_payload["result"]["plot"]))

            result_response = self.client.get(response["Location"])
            self.assertEqual(result_response.status_code, 200)
            self.assertNotContains(result_response, "Figuras auxiliares")

    def test_truncation_error_is_reported_for_integration_methods(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "trapecio-simple"}),
            data={
                "fx": "sin(x)",
                "a": 0.0,
                "b": 3.14159,
                "xi": 1.5707963267948966,
                "seed": 42,
                "precision": 6,
            },
        )
        self.assertEqual(response.status_code, 302)

        result_response = self.client.get(response["Location"])
        self.assertEqual(result_response.status_code, 200)
        self.assertContains(result_response, "error de truncamiento")
        self.assertContains(result_response, "xi")

    def test_composite_trapezoid_handles_removable_singularity(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "trapecio-compuesta"}),
            data={
                "fx": r"\ln(x+1)/x",
                "a": 0.0,
                "b": 1.0,
                "n": 4,
                "xi": 0.5,
                "seed": 42,
                "precision": 6,
            },
        )
        self.assertEqual(response.status_code, 302)

        runs = self.client.session.get("simulation_runs", {})
        payload = runs[next(reversed(runs))]
        integral_value = payload["result"]["summary"]["integral aproximada"]
        self.assertTrue(integral_value == integral_value)
        self.assertNotEqual(str(integral_value).lower(), "nan")

    def test_monte_carlo_simple_reports_statistics_and_distribution_plot(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "monte-carlo-simple"}),
            data={
                "fx": "sin(x)",
                "a": 0.0,
                "b": 3.14159,
                "samples": 1500,
                "confidence_level": 0.95,
                "seed": 42,
                "precision": 6,
            },
        )
        self.assertEqual(response.status_code, 302)

        runs = self.client.session.get("simulation_runs", {})
        payload = runs[next(reversed(runs))]
        summary = payload["result"]["summary"]
        for metric in [
            "media muestral",
            "varianza",
            "desviacion estandar",
            "error estandar",
            "nivel de confianza",
            "intervalo de confianza",
        ]:
            self.assertIn(metric, summary)

        aux_plots = payload["result"].get("auxiliary_plots", [])
        self.assertGreaterEqual(len(aux_plots), 1)
        self.assertIn("Histograma de valores + normal estandar", aux_plots[0].get("title", ""))
        self.assertIn("plotly-graph-div", aux_plots[0].get("plot", ""))
        self.assertIn("normal estandar", aux_plots[0].get("plot", ""))

    def test_root_finding_methods_include_original_function_plot(self):
        runs_to_check = [
            (
                "biseccion",
                {"fx": "x**3 - x - 2", "a": 1.0, "b": 2.0, "tol": 1e-6, "max_iter": 50, "precision": 6},
            ),
            (
                "newton-raphson",
                {
                    "fx": "x**3 - x - 2",
                    "dfx": "3*x**2 - 1",
                    "x0": 1.5,
                    "tol": 1e-6,
                    "max_iter": 50,
                    "precision": 6,
                },
            ),
            (
                "aitken-delta-cuadrado",
                {"gx": "cos(x)", "x0": 0.5, "tol": 1e-6, "max_iter": 50, "precision": 6},
            ),
        ]

        for slug, payload in runs_to_check:
            response = self.client.post(reverse("simulations:run", kwargs={"slug": slug}), data=payload)
            self.assertEqual(response.status_code, 302)

            runs = self.client.session.get("simulation_runs", {})
            self.assertTrue(runs)
            stored_payload = runs[next(reversed(runs))]
            aux_plots = stored_payload["result"].get("auxiliary_plots", [])
            self.assertGreaterEqual(len(aux_plots), 1)
            self.assertIn("plotly-graph-div", str(aux_plots[0].get("plot", "")))

            result_response = self.client.get(response["Location"])
            self.assertEqual(result_response.status_code, 200)
            self.assertContains(result_response, "Figuras auxiliares")

