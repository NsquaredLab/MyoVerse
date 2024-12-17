import importlib

import pytest
from pathlib import Path


class TestModelAPI:
    def test_model_names_raul_net(self):
        # get a list of all model names
        model_data = [
            (p.parent.name, p.name.replace(".py", ""))
            for p in Path("../../myoverse/models/definitions/raul_net").rglob("*.py")
            if "__init__" not in p.name
        ]
        model_data = sorted(model_data, key=lambda x: x[1])

        # check if the model names are correct
        for model_type, model_name in model_data:
            try:
                imported_module = importlib.import_module(
                    f"myoverse.models.definitions.raul_net.{model_type}.{model_name}"
                )

                assert hasattr(
                    imported_module, f"RaulNetV{model_name.split('_')[0][1:]}"
                )

            except ModuleNotFoundError:
                pytest.fail(
                    f"Could not import myoverse.models.definitions.raul_net.{model_type}.{model_name}"
                )
