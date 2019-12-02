from collections import OrderedDict
import natsort
from itertools import combinations

# Ia should always be first
# TODO can sntypes change ?
SNTYPES = OrderedDict(
    {
        "101": "Ia",
        "120": "IIP",
        "121": "IIn",
        "122": "IIL1",
        "123": "IIL2",
        "132": "Ib",
        "133": "Ic",
    }
)

LIST_FILTERS = natsort.natsorted(["r", "g", "i", "z"])
LIST_FILTERS_COMBINATIONS = []
for i in range(1, len(LIST_FILTERS) + 1):
    LIST_FILTERS_COMBINATIONS += combinations(LIST_FILTERS, i)
LIST_FILTERS_COMBINATIONS = natsort.natsorted(
    ["".join(e) for e in LIST_FILTERS_COMBINATIONS]
)
FILTER_DICT = OrderedDict()
INVERSE_FILTER_DICT = OrderedDict()
for i, e in enumerate(LIST_FILTERS_COMBINATIONS):
    FILTER_DICT[e] = i
    INVERSE_FILTER_DICT[i] = e

# TODO moved from data utils
OFFSETS = [-2, -1, 0, 1, 2]
# OOD_TYPES = ["random", "reverse", "shuffle", "sin"]
OOD_TYPES = []
OFFSETS_STR = ["-2", "-1", "", "+1", "+2"]
