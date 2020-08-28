#!/usr/bin/env python3
import functools
import logging

logger = logging.getLogger('experiment')
logger.setLevel(logging.INFO)

class Experiment:
    def __init__(self, name, entities=None, config=None, setups=None, transforms=None):
        self.name = name
        self.entities = {} if entities is None else entities
        self.config = {} if config is None else config
        self.transforms = {} if transforms is None else transforms
        self.setups = {} if setups is None else setups

    def entity(self, _obj=None, *, name=None):
        """Decorator for registering a class or function
        Can optionally be called with a name argument, otherwise the
        object/function name is used.
        """
        if _obj is None:  # name argument
            def entity_register(_obj):
                self.entities[name] = _obj
                return _obj
            return entity_register
        else:             # No argments
            self.entities[_obj.__name__] = _obj
            return _obj

    def add_arguments(self, _fn):
        _fn()
        return _fn

    def setup(self, name):
        def setup_register(setup_fn):
            self.setups[name] = setup_fn
            return setup_fn
        return setup_register

    def transform(self, name):
        def transform_register(transform):
            self.transforms[name] = transform
            return transform
        return transform_register

    def get_and_configure(self, name):
        setup_fn = self.setups[name]
        return setup_fn(self.config)

    def merge(self, d1, d2, other=""):
        intersect = d1.keys() & d2.keys()
        if len(intersect) > 0:
            logger.warning(f"When merging {self.name} and {other}, item(s) {intersect} found multiple times.")
        d1.update(d2)
        return d1

    def __add__(self, exp):
        self.entities = self.merge(self.entities, exp.entities, other=exp.name)
        self.config = self.merge(self.config, exp.config, other=exp.name)
        self.transforms = self.merge(self.transforms, exp.transforms, other=exp.name)
        return self

# Instantiate once
ex = Experiment('base')

def test_experiment():
    ex = Experiment('test', config={'entity1': {'a': 1, 'b': 2}})
    @ex.config('new')
    def cfg():
        return {'b': 4, 'c': 2}
    print(ex.config)

    ex2 = Experiment('test2', config={'entity1': {'a': 3, 'c': 4}})
    ex += ex2
    print(ex.config)

def test_entity_registration():
    ex = Experiment('test')
    @ex.entity
    class MyClass:
        def __init__(self):
            self.greeting = "hello"

    @ex.entity(name="custom_name")
    class MyClass2:
        def __init__(self):
            self.greeting = "sup"

    print(ex.entities)

if __name__ == "__main__":
    from pdb import set_trace
    test_experiment()
    test_entity_registration()



"""
Features:
Maintain entire experiment config in a single object.
- Default arguments
Allow config to be specified from source file where the object is defined.
Object is configured flexibly: instead of setting a bunch of attributes, give
model access to config at runtime.

"""
