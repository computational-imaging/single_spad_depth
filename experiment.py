#!/usr/bin/env python3
import functools
import logging

logger = logging.getLogger('experiment')
logger.setLevel(logging.INFO)

class Experiment:
    def __init__(self, name, entities=None, configs=None, transforms=None):
        self.name = name
        self.entities = {} if entities is None else entities
        self.configs = {} if configs is None else configs
        self.transforms = {} if transforms is None else transforms

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

    def transform(self, name):
        def transform_register(transform):
            self.transforms[name] = transform
            return transform
        return transform_register

    def merge(self, d1, d2, other=""):
        intersect = d1.keys() & d2.keys()
        if len(intersect) > 0:
            logger.warning(f"When merging {self.name} and {other}, item(s) {intersect} found multiple times.")
        d1.update(d2)
        return d1

    def __add__(self, exp):
        self.entities = self.merge(self.entities, exp.entities, other=exp.name)
        self.configs = self.merge(self.configs, exp.configs, other=exp.name)
        self.transforms = self.merge(self.transforms, exp.transforms, other=exp.name)
        return self.__class__(ntities, configs, transforms)

# Instantiate once
ex = Experiment('base')

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
