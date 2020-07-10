import setuptools

setuptools.setup(
    name="supernnova",
    version="1.3",
    author="Anais Moller and Thibault de Boissiere",
    author_email="anais.moller@clermont.in2p3.fr",
    description="framework for Bayesian, Neural Network based supernova light-curve classification",
    url="https://github.com/supernnova/SuperNNova",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require={
        "docs": [
            "sphinx >= 1.4",
            "sphinx_rtd_theme",
            "sphinx-autobuild",
            "sphinxcontrib-napoleon",
        ]
    },
)
