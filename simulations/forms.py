from django import forms

from .registry import SimulationSpec


_LATEX_HELP_SUFFIX = "Escriba LaTeX sin $...$; la vista previa se actualiza al escribir."


class DynamicSimulationForm(forms.Form):
    @staticmethod
    def _build_parameter_field(parameter, common_kwargs: dict) -> forms.Field:
        if parameter.kind == "int":
            return forms.IntegerField(
                min_value=parameter.min_value,
                max_value=parameter.max_value,
                **common_kwargs,
            )
        if parameter.kind == "float":
            return forms.FloatField(
                min_value=parameter.min_value,
                max_value=parameter.max_value,
                **common_kwargs,
            )
        raise ValueError(f"Tipo de parametro no soportado: {parameter.kind}")

    def __init__(self, spec: SimulationSpec, *args, **kwargs):
        self.spec = spec
        super().__init__(*args, **kwargs)

        for expression in spec.expressions:
            self.fields[expression.name] = forms.CharField(
                label=expression.label,
                initial=expression.default,
                help_text=(f"{expression.help_text} {_LATEX_HELP_SUFFIX}").strip(),
                widget=forms.TextInput(
                    attrs={
                        "placeholder": expression.default,
                        "data-math-input": "true",
                    }
                ),
            )

        for parameter in spec.parameters:
            common_kwargs = {
                "label": parameter.label,
                "initial": parameter.default,
                "help_text": parameter.help_text,
            }
            if parameter.name == "xi":
                common_kwargs["required"] = False
            self.fields[parameter.name] = self._build_parameter_field(parameter, common_kwargs)

        if "precision" not in self.fields:
            self.fields["precision"] = forms.IntegerField(
                label="Precision (decimales)",
                initial=6,
                min_value=0,
                max_value=12,
                help_text="Cantidad de decimales usada en los calculos y resultados.",
            )

