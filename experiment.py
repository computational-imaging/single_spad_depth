#!/usr/bin/env python3
import functools
import logging

logger = logging.getLogger('experiment')
logger.setLevel(logging.INFO)

class Experiment:
    def __init__(self, name, entities=None, configs=None):
        self.name = name
        self.entities = {} if entities is None else entities
        self.configs = {} if configs is None else configs

    def entity(self, name):
        """Decorator for registering a class or function as a model
        """
        def entity_register(obj):
            self.entities[name] = obj
            return obj
        return entity_register

    def config(self, name):
        def config_register(cfg_fn):
            cfg = cfg_fn()
            self.configs[name] = cfg
            return cfg_fn
        return config_register

    @staticmethod
    def merge(cfg1, cfg2, key=""):
        intersect = cfg1.keys() & cfg2.keys()
        if len(intersect) > 0:
            logger.warning(f"Config item(s) {intersect} found multiple times. ({key})")
        cfg1.update(cfg2)
        return cfg1

    def __add__(self, exp):
        self.entities.update(exp.entities)
        self.configs = Experiment.merge(self.configs, exp.configs, key=exp.name)
        return self

def test_experiment():
    ex = Experiment('test', configs={'entity1': {'a': 1, 'b': 2}})
    @ex.config('new')
    def cfg():
        return {'b': 4, 'c': 2}
    print(ex.configs)

    ex2 = Experiment('test2', configs={'entity1': {'a': 3, 'c': 4}})
    ex += ex2
    print(ex.configs)

if __name__ == "__main__":
    test_experiment()




"""
Features:
Maintain entire experiment config in a single object.
- Default arguments
Allow config to be specified from source file where the object is defined.
Object is configured flexibly: instead of setting a bunch of attributes, give
model access to config at runtime.

"""
