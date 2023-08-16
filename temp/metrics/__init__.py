
import os
import glob
import importlib

from metrics.core.metric import MetricClass

metric_files = glob.glob("*/metrics/metric_*.py")
loaded_metric_classes = []
for metric_file in metric_files:
    module_name = os.path.splitext(metric_file)[0].replace(os.path.sep,'.').replace("temp.","")
    module = importlib.import_module(module_name)

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if (
            isinstance(attribute,type) 
            and issubclass(attribute, MetricClass)
            and attribute != MetricClass
        ):
            loaded_metric_classes.append(attribute)

for cls in loaded_metric_classes:
    print("Imported metric:", cls.__name__)