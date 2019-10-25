import matplotlib.pylab as plt

# Plotting styles
ALL_COLORS = [
    "maroon",
    "darkorange",
    "royalblue",
    "indigo",
    "black",
    "maroon",
    "darkorange",
    "royalblue",
    "indigo",
]
BI_COLORS = ["darkorange", "royalblue"]
CONTRAST_COLORS = ["darkorange", "indigo"]
MARKER_DIC = {"randomforest": "o", "vanilla": "s"}
FILL_DIC = {"None": "none", "zpho": "bottom", "zspe": "full"}
MARKER_LIST = ["o", "o", "v", "v", "^", "^",
               ">", ">", "<", "<", "s", "s", "D", "D"]
CMAP = plt.cm.YlOrBr
LINE_STYLE = ["-", "-", "-", "-", ":", ":", ":", ":", "-.", "-.", "-.", "-."]
FILTER_COLORS = {"z": "maroon", "i": "darkorange",
                 "r": "royalblue", "g": "indigo", "u": "purple", "Y": "red"}
PATTERNS = ["", "", "", "", "", ".", ".", ".", "."]


def get_model_visualization_name(model_name):

    if "bayesian" in model_name or "BBB" in model_name:
        return "BBB RNN"
    if "variational" in model_name:
        return "Variational RNN"
    if "vanilla" in model_name or "baseline" in model_name:
        return "Baseline RNN"
    if "forest" in model_name:
        return "Random Forest"
