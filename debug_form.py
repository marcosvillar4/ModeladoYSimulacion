                                                   #!/usr/bin/env python
import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from simulations.services import register_default_simulations, get_simulation
from simulations.forms import DynamicSimulationForm

register_default_simulations()
spec = get_simulation('punto-fijo')

print("Parameters registered:")
for p in spec.parameters:
    print(f"  - {p.name}: {p.label}")

print("\nForm fields:")
form = DynamicSimulationForm(spec=spec)
for field_name in form.fields:
    print(f"  - {field_name}")

