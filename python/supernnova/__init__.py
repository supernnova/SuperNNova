import importlib.metadata

name = "supernnova"
try:
    __version__ = importlib.metadata.version("supernnova")
except importlib.metadata.PackageNotFoundError:
    # Package is being used directly from source without being installed.
    # This happens when running scripts with sys.path pointing to python/.
    __version__ = "unknown"
