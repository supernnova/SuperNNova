.. _CondaConfigurations:

Environment configuration
=============================

Conda virtual env
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The preferred option to setup your environment is through conda environment as follows:

.. code::

	conda create --name <env> --file conda_env.txt

example configuration files for linux-64 (cpu and gpu) and osx-64 are provided in ``SuperNNova/env``.

.. _DockerConfigurations:

Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use docker:

- Install docker: `Docker`_.

Create a docker image:

.. code::

    cd env && make {image}

where ``image`` is one of ``cpu`` or ``gpu`` (for cuda 9.) or ``gpu10`` (for cuda 10.)

- This images contains all of this repository's dependencies.
- Image construction will typically take a few minutes

Enter docker environment by calling:

.. code::

    python launch_docker.py --image <image> --dump_dir </path/to/data>

- Add ``--image image`` where image is ``cpu`` or ``gpu`` (for cuda 9.) or ``gpu10`` (for cuda 10.)
- Add ``--dump_dir /path/to/data`` to mount the folder where you stored the data (see :ref:`DataStructure`) into the container. If unspecified, will use the default location (i.e. ``snndump``)

This will launch an interactive session in the docker container, with zsh support.

.. _Docker: https://docs.docker.com/install/linux/docker-ce/ubuntu/