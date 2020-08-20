#!/usr/bin/env python3
import functools

REGISTRY = {}

def register(name):
    """Decorator for registering a class or function as a model
    """
    def decorator_register(func):
        REGISTRY[name] = func
        return func
