from setuptools import setup

setup(
    name="my_package",
    extras_require={
        "docs": [
            "sphinx >= 1.4",
            "sphinx_rtd_theme",
            "sphinx-autobuild",
            "sphinxcontrib-napoleon",
        ]
    },
)
