from collections import OrderedDict
import natsort
from itertools import combinations

# If observations are taken within 0.33 days of each other, they get assigned the same time
MIN_DT = 0.33

# Ia should always be first
SNTYPES = OrderedDict(
    {
        "0": "Ia",
        "101": "Ia",
        "120": "IIP",
        "121": "IIn",
        "122": "IIL1",
        "123": "IIL2",
        "132": "Ib",
        "133": "Ic",
        "1": "Ibc",
        "5": "Ibc",
        "6": "Ibc",
        "7": "Ibc",
        "8": "Ibc",
        "9": "Ibc",
        "10": "Ibc",
        "11": "Ibc",
        "13": "Ibc",
        "14": "Ibc",
        "16": "Ibc",
        "18": "Ibc",
        "22": "Ibc",
        "23": "Ibc",
        "29": "Ibc",
        "45": "Ibc",
        "28": "Ibc",
        "2": "II",
        "3": "II",
        "4": "II",
        "12": "II",
        "15": "II",
        "17": "II",
        "19": "II",
        "20": "II",
        "21": "II",
        "24": "II",
        "25": "II",
        "26": "II",
        "27": "II",
        "30": "II",
        "31": "II",
        "32": "II",
        "33": "II",
        "34": "II",
        "35": "II",
        "36": "II",
        "37": "II",
        "38": "II",
        "39": "II",
        "40": "II",
        "41": "II",
        "42": "II",
        "43": "II",
        "44": "II",
        "-9": "unknown"
    
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

OFFSETS = [-2, -1, 0, 1, 2]
# OOD_TYPES = ["random", "reverse", "shuffle", "sin"]
OOD_TYPES = []
OFFSETS_STR = ["-2", "-1", "", "+1", "+2"]
