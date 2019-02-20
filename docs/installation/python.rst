.. _DockerConfigurations:

Environment configuration
=============================


Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The preferred option to setup your environment is through docker:

- Install docker: `Docker`_.
- For GPU support, install nvidia docker: `NVDocker`_.

Create a docker image:

.. code::

    cd docker && make {device}

where ``device`` is one of ``cpu`` or ``gpu``

- This images contains all of this repository's dependencies.
- Image construction will typically take a few minutes

Enter docker environment by calling:

.. code::

    python launch_docker.py

- Add ``--use_cuda`` to launch a GPU-supported container
- Add ``--dump_dir /path/to/data`` to mount the folder where you stored the data (see :ref:`DataStructure`) into the container. If unspecified, will use the default location (i.e. ``snndump``)

This will launch an interactive session in the docker container, with zsh support.

.. _CondaConfigurations:


Conda virtual env
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, one can setup a conda environment  as follows:

.. code::

	conda create --name <envname> --file conda_env.txt

you can activate this envirionment with ``source conda <envname>``. Else simply install in your current environment ``conda install --file conda_env.txt``.

example configuration files for linux-64 (cpu and gpu) and osx-64 are provided in ``SuperNNova/env``.

.. _Docker: https://docs.docker.com/install/linux/docker-ce/ubuntu/
.. _NVDocker: https://github.com/NVIDIA/nvidia-docker