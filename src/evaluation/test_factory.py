from enum import Enum

from ml_collections import ConfigDict

from src.evaluation.tests.Interpolation_test import Inter_Extrapolation_Test
from src.evaluation.tests.drag_coefficient_test import Drag_Coefficient_Test
from src.evaluation.tests.parameter_comparison_test import Parameter_Comparison_Test
from src.evaluation.tests.raf_30_test import Raf30_test
from src.evaluation.tests.sampling_speed_test import Sampling_Speed_Test


class TestType(Enum):
    SAMPLING_SPEED = 1
    INTERPOLATION = 2
    EXTRAPOLATION = 3
    DRAG_COEFFICIENT = 4
    PARAMETER_COMPARISON = 5

class Test_Factory():
    def __init__(self):
        pass

    @staticmethod
    def get_evaluation_test(test_type: TestType, config:ConfigDict):
        if test_type == TestType.INTERPOLATION :
            return Inter_Extrapolation_Test(config)
        elif test_type == TestType.EXTRAPOLATION :
            return Inter_Extrapolation_Test(config)
        elif test_type == TestType.SAMPLING_SPEED :
            return Raf30_test(config)
        elif test_type == TestType.DRAG_COEFFICIENT :
            return Drag_Coefficient_Test(config)
        elif test_type == TestType.SAMPLING_SPEED :
            return Sampling_Speed_Test(config)
        elif test_type == TestType.PARAMETER_COMPARISON :
            return Parameter_Comparison_Test(config)
        return None
