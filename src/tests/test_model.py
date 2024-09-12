import torch
import pytest
from src.models import model_select

@pytest.fixture
def load_model(model_name: str):
    model_config, model = model_select.get_model(model_name)
    model.eval()
    return model

def test_model_output_shape(load_model):
    input_shape = (1, 3, 32, 32)
    expected_output_shape = (1, 3, 32, 32)

    input_tensor = torch.randn(input_shape)
    model = load_model("swin")
    with torch.no_grad():
        output_tensor = model(input_tensor)

    assert output_tensor.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output_tensor.shape}"