import unittest

from models import (CadeneModel, 
                    TorchVisionModel, 
                    EfficientNetModel,
                    Detectron2Model,
                    YOLOv3Model,
                    EfficientDetModel)
from models import get_model
from util import Args

MODELS = [cls.__name__ for cls in
          CadeneModel.__subclasses__() + \
          TorchVisionModel.__subclasses__() + \
          EfficientNetModel.__subclasses__() + \
          YOLOv3Model.__subclasses__() + \
          Detectron2Model.__subclasses__()]


class TestModelMeta(type):
    def __new__(mcs, name, bases, dict):
        def gen_test(model_name, pretrained=False, num_classes=None):
            def test_model(self):
                args = Args({"model": model_name,
                             "pretrained": pretrained,
                             "gpus": None})
                if num_classes is not None:
                    args['num_classes'] = num_classes
                model = get_model(args)
                self.assertIsNotNone(model)

            return test_model

        for model in MODELS:
            dict[f"test_{model}_num_classes"] = gen_test(model, num_classes=1)

        return type.__new__(mcs, name, bases, dict)


class TestModel(unittest.TestCase,
                metaclass=TestModelMeta):
    pass


if __name__ == "__main__":
    unittest.main(verbosity=0)
