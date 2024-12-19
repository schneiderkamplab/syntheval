
import glob
import importlib
import os
from os.path import basename, dirname, isfile, join

from .core.metric import MetricClass


def load_metrics():
    
    loaded_metrics = {}
    metric_files = glob.glob(join(dirname(__file__),"**/metric_*.py"),recursive=True)

    for metric_file in metric_files:
        module_name = os.path.splitext(metric_file)[0].replace(dirname(__file__),'syntheval.metrics').replace(os.path.sep,'.')

        module = importlib.import_module(module_name)

        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                isinstance(attribute,type)
                and issubclass(attribute, MetricClass)
                and attribute != MetricClass
            ):
                loaded_metrics[attribute.name()] = attribute
    return loaded_metrics
