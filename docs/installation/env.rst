.. _CondaConfigurations:

Environment configuration
=============================

Conda virtual env
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The preferred option to setup your environment is through conda environment.

The setup of environment can be done in two steps. Navigate to the root of the downloaded repository, then

1. Create a conda environment:

.. code-block:: bash

    conda env create -f env/conda_env.yml

or 

.. code-block:: bash

    conda env create -f env/conda_gpu_env.yml

if you want to install ``PyTorch`` with cuda support.

2. Activate the conda environment:

.. code-block:: bash

    conda activate supernnova

or 

.. code-block:: bash

    conda activate supernnova-cuda

if you create environment from "conda_gpu_env.yml".

3. A python project management tool ``poetry`` is installed via the above steps. Verify it and install python dependencies for this project:

.. code-block:: bash

    which poetry # should print <conda env>/bin/poetry
    poetry install

For developers (including testing local documentation), please refer to :ref:`dev-python-env`.


.. _DockerConfigurations:

Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use docker:

- Install docker: `Docker`_.

Create a docker image:

.. code::

    make {image}

where ``image`` is one of ``cpu`` or ``gpu`` (for the latest supported CUDA version; currently 12.3.1) or ``gpu9`` (for cuda 9.0)

- This image contains all of this repository's dependencies.
- Image construction will typically take a few minutes

Enter docker environment by calling:

.. code::

    python env/launch_docker.py --image <image> --dump_dir </path/to/data>

- Add ``--image image`` where image is ``cpu`` or ``gpu`` (latest version) or ``gpu9`` (for cuda 9)
- Add ``--dump_dir /path/to/data`` to mount the folder where you stored the data (see :ref:`DataStructure`) into the container. If unspecified, will use the default location (i.e. ``snndump``)

This will launch an interactive session in the docker container, with zsh support.

.. _Docker: https://docs.docker.com/install/linux/docker-ce/ubuntu/
