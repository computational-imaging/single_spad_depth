#!/usr/bin/env python3

from abc import ABCMeta

class MDE:
    def __init__(self):
        pass

    def predict(self, rgb):
        """
        Given an RGB image in [0, 1], return a depth map of the same size.
        """
        raise NotImplementedError
