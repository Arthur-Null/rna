#coding=utf-8

import math

def entropyOfList(ls):
    """
    Given a list of some items, compute entropy of the list
    The entropy is sum of -p[i]*log(p[i]) for every unique element i in the list, and p[i] is its frequency
    """
    elements = {}
    for e in ls:
        elements[e] = elements.get(e, 0) + 1
    length = float(len(ls))
    return sum([-v/length*math.log(v/length) for v in list(elements.values())])



