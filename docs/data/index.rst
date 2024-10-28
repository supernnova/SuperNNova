.. _DataStructure:

Data walkthrough
=========================

Recommended code organization structure:

.. code::

    ├── snndump        (to save the data)
    │
    ├── SuperNNova
    │   ├── python/supernnova
    │   ├── env
    │   ├── docs
    │   ├── tests


**To build the database:**

- Ensure you have raw data saved to ``{raw_dir}/raw``
- The default settings assume the raw data and fits are saved to ``snndump/raw``
- You can save the data in any folder, but you then have to specify the ``dump_dir`` with the ``--dump_dir XXX`` command.
- You can specify a different place where the raw data is using ``--raw_dir XXX`` command.
- You can specify a different place where the fits to data is using ``--fits_dir XXX`` command.


Activate the environment
-------------------------------

**Either use docker**

.. code-block:: bash

    cd env && python launch_docker.py (--use_cuda optional)

**Or activate your conda environment**

.. code-block:: bash

    source activate <conda_env_name>

Create a database
-------------------------------
To create a database, you can use ``snn make_data`` with valid options:

.. code-block:: bash

    snn make_data [option]

A list of valid options can be shown by using the ``--help`` flag:

.. code-block:: bash

    snn make_data --help

.. code-block:: none

    usage: snn make_data [options]

    optional arguments:
    --config_file                  YML config file
    --data_fraction                Fraction of data to use
    --data_testing                 Create database with only validation set
    --data_training                Create database with mostly training set of 99.5%%
    --debug                        Debug database creation: one file processed only
    ... ...

Below, we detail several of the most frequently used approaches to create a database. You can also use a YAML file to specify option arguments. Please see :ref:`UseYaml` for more information.

1. Create a debugging database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    snn make_data --dump_dir tests/dump --raw_dir tests/raw 

- This creates a database for a very small subset of all available data
- This is intended for debugging purposes (training, validation can run very fast with this small database)
- The database is saved to the specified ``tests/dump/processed``


2. Create a database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    snn make_data --dump_dir <path/to/full/database/> --raw_dir <path/to/raw/data/> 


- You **DO NEED** to download the raw data for this database or point where your data is.
- This creates a database for all the available data with 80/10/10 train/validate/test splits. 
- Splits can be changed using ``--data_training`` (use data only for raining and validation) or ``--data_testing`` (use data only for testing) commands. For yaml just add ``data_training: True`` or ``data_testing: True``.
- The database is saved to the specified ``dump_dir``, in the ``processed`` subfolder.
- There is no need to specify salt2fits file to make the dataset. It can be used if available but it is not needed ``--fits_dir <empty/path/>``.
- Raw data can be in csv format with columns:
- `` DES_PHOT.csv``: SNID,MJD, FLUXCAL, FLUXCALERR, FLT 
- `` DES_HEAD.csv``: SNID, PEAKMJD, HOSTGAL_PHOTOZ, HOSTGAL_PHOTOZ_ERR, HOSTGAL_SPECZ, HOSTGAL_SPECZ_ERR, SIM_REDSHIFT_CMB, SIM_PEAKMAG_z, SIM_PEAKMAG_g, SIM_PEAKMAG_r, SIM_PEAKMAG_i, SNTYPE.


3. Create a database for testing a trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is how to create a database with only lightcurves to evaluate.

.. code-block:: bash

    snn make_data --dump_dir <path/to/save/database/> --data_testing  --raw_dir <path/to/raw/data/> 

Note that:
- using ``--data_testing`` option will generate a 100% testing set (see below for more details).
**Using command yaml:** modify the configuration file with ``data_testing: True``.


4. Create a database using some SNIDs for testing and the rest for training and validating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is how to create a database using a list of SNIDs for testing. 

.. code-block:: bash

    snn make_data --dump_dir <path/to/save/database/> --raw_dir <path/to/raw/data/> --testing_ids <path/to/ids/file>

You can provide the SNIDs in ``.csv`` or ``.npy`` format. The ``.csv`` must contain a column ``SNID``.


5. Create a database with photometry limited to a time window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Photometric measurements may span over a larger time range than the one desired for classification. For example, a year of photometry is much larger than the usual SN timespan. Therefore, it may be desirable to just use a subset of this photometry (observed epochs cuts). To do so:

.. code-block:: bash

    snn make_data --dump_dir <path/to/save/database/> --raw_dir <path/to/raw/data/>  --photo_window_files <path/to/csv/with/peakMJD> --photo_window_var <name/of/variable/in/csv/to/cut/on> --photo_window_min <negative/int/indicating/days/before/var> --photo_window_max <positive/int/indicating/days/after/var> 

6. Create a database with different survey
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The default filter set is the one from the Dark Energy Survey Supernova Survey ``g,r,i,z``. If you want to use your own survey, you'll need to specify your filters.

.. code-block:: bash

    snn make_data --dump_dir <path/to/save/database/> --raw_dir <path/to/raw/data/>  --list_filters <your/filters>

e.g. ``--list_filters g r``. 

7. Use a different redshift label
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The default redshift label is either ``HOSTGAL_SPECZ``/``HOSTGAL_PHOTOZ`` (with option ``zspe``/``zpho``). If you want to use your own label, you'll need to specify it. Beware, this will override also ``SIM_REDSHIFT_CMB`` used for the title of plotted light-curves.

.. code-block:: bash

    snn make_data --dump_dir <path/to/save/database/> --raw_dir <path/to/raw/data/>  --redshift_label <your/label>

e.g. ``--redshift_label REDSHIFT_FINAL``. 

8. Use a different sntype label
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The default sntype label is ``SNTYPE``. If you want to use your own label, you'll need to specify it. 

.. code-block:: bash

    snn make_data --dump_dir <path/to/save/database/> --raw_dir <path/to/raw/data/>  --sntype_var <your/label>

e.g. ``--redshift_label SIM_SNTYPE``. 

9. Mask photometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The default is to use all available photometry for classification. However, we support masking photometric epochs with a power of two mask. Any combination of these power of two integers, and with other numbers, will be eliminated from the database.

.. code-block:: bash

    snn make_data --dump_dir <path/to/save/database/> --raw_dir <path/to/raw/data/>  --phot_reject <your/label> --phot_reject_list <list/to/reject>

e.g. ``--phot_reject PHOTFLAG --phot_reject_list 8 16 32 64 128 256 512``. 


10. Add another training variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You may want to add another feature for training and classification from the metadata (HEAD for .fits)

.. code-block:: bash

    snn make_data --dump_dir <path/to/save/database/> --raw_dir <path/to/raw/data/>  --additional_train_var <additional_column_name>

e.g. ``--additional_train_var MWEBV``. 


Under the hood
-------------------------------

Preparing data splits
~~~~~~~~~~~~~~~~~~~~~~

We first compute the data splits:

- By default the HEAD FITS/csv files are analyzed to compute 80/10/10 train/test/val splits.
- You can change if the database contains 99.5/0.5/0.5 train/test/val splits using ``--data_training`` command. 
- You can change if the database contains 0/0/100 train/test/val splits using ``--data_testing`` command. Beware, this option has other consequences.
- The splits are different for the salt/photometry datasets
- The splits are different depending on the classification target
- We downsample the dataset so that for a given classification task, all classes have the same cardinality
- The supernova/light-curve types supported can be changed using ``--sntypes``. Default contains 7 classes. If a class is not given as input in ``--sntypes``, it will be assigned to the last available tag. If a 'Ia' exists in  provided ``--sntypes``, this will be taken as tag 0 in data splits, else the first class will be used.

Preprocessing
~~~~~~~~~~~~~~

We then pre-process each FITS/csv file

- Join column from header files
- Select columns that will be useful later on
- Compute SNID to tag each light curve
- Compute delta times between measures
- Removal of delimiter rows


Pivot
~~~~~~~~~~~~~~

We then pivot each preprocessed file: we will group time-wise close observations on the same row
and each row in the dataframe will show a value for each of the flux and flux error column

- All observations within 8 hours of each other are assigned the same MJD
- Results are cached with pickle for faster loading


HDF5
~~~~~~~~~~~~~~

The processed database is saved to ``dump_dir/processed`` in HDF5 format for convenient use
in the ML pipeline

The HDF5 file is organized as follows:

.. code::

    ├── data                            (variable length array to store time series)
    │
    │
    ├── dataset_photometry_2classes     (0: train set, 1: valid set, 2: test set, -1: not used)
    ├── dataset_photometry_7classes     (0: train set, 1: valid set, 2: test set, -1: not used)
    │
    ├── target_photometry_2classes      (integer between 0 and 1, included)
    ├── target_photometry_7classes      (integer between 0 and 6, included)
    │
    │
    ├── features                        (array of str: feature names to be used)
    ├── normalizations
    │   ├── FLUXCAL_g
    │        ├── min
    │        ├── mean                    Normalization coefficients for that feature
    │        ├── std
    │    ...
    ├── normalizations_global
    │   ├── FLUXCAL
    │       ├── min
    │       ├── mean                    Normalization coefficients for that feature
    │       ├── std                     In this scheme, the coefficients are shared between fluxes and flux errors
    │   ...
    │
    ├── SNID                            The ID of the lightcurve
    ├── PEAKMJD                         The MJD value at which a lightcurve reaches peak light
    ├── SNTYPE                          The type of the lightcurve (120, 121...)
    │
    ...                                 (Other metadata / features about lightcurves)


The features used for classification are the following:

- **FLUXCAL_g** (flux)
- **FLUXCAL_i** (flux)
- **FLUXCAL_r** (flux)
- **FLUXCAL_z** (flux)
- **FLUXCALERR_g** (flux error)
- **FLUXCALERR_i** (flux error)
- **FLUXCALERR_r** (flux error)
- **FLUXCALERR_z** (flux error)
- **delta_time** (time elapsed since previous observation in MJD)
- **HOSTGAL_PHOTOZ** (photometric redshift)
- **HOSTGAL_PHOTOZ_ERR** (photometric redshift error)
- **HOSTGAL_SPECZ** (spectroscopic redshift)
- **HOSTGAL_SPECZ_ERR** (spectroscopic redshift eror)
- **g** (boolean flag indicating which band is present at a specific time step)
- **gi** (boolean flag indicating which band is present at a specific time step)
- **gir** (boolean flag indicating which band is present at a specific time step)
- **girz** (boolean flag indicating which band is present at a specific time step)
- **giz** (boolean flag indicating which band is present at a specific time step)
- **gr** (boolean flag indicating which band is present at a specific time step)
- **grz** (boolean flag indicating which band is present at a specific time step)
- **gz** (boolean flag indicating which band is present at a specific time step)
- **i** (boolean flag indicating which band is present at a specific time step)
- **ir** (boolean flag indicating which band is present at a specific time step)
- **irz** (boolean flag indicating which band is present at a specific time step)
- **iz** (boolean flag indicating which band is present at a specific time step)
- **r** (boolean flag indicating which band is present at a specific time step)
- **rz** (boolean flag indicating which band is present at a specific time step)
- **z**  (boolean flag indicating which band is present at a specific time step)