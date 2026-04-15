from django.test import TestCase
from django.urls import reverse

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

    def test_run_fixed_point_redirects_to_result(self):
        response = self.client.post(
            reverse("simulations:run", kwargs={"slug": "punto-fijo"}),
            data={"gx": "cos(x)", "x0": 0.5, "tol": 1e-6, "max_iter": 50, "seed": 42, "precision": 6},
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
            data={"gx": r"\frac{1}{2}x", "x0": 2.0, "tol": 1e-6, "max_iter": 50, "seed": 42, "precision": 6},
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
                {"fx": "sin(x)", "a": 0.0, "b": 3.14159, "n": 6, "seed": 42, "precision": 6},
            ),
            (
                "simpson-13-compuesta",
                {"fx": "sin(x)", "a": 0.0, "b": 3.14159, "n": 6, "seed": 42, "precision": 6},
            ),
            (
                "regla-rectangulo",
                {"fx": "sin(x)", "a": 0.0, "b": 3.14159, "n": 8, "seed": 42, "precision": 6},
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

