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
The default sntype label is ``SNTYPE``. If you want to use your own label, you'll need to specify it and provide it in the `HEAD` file.

.. code-block:: bash

    snn make_data --dump_dir <path/to/save/database/> --raw_dir <path/to/raw/data/>  --sntype_var <your/label>

e.g. ``--sntype_var MYTYPE``.

**Note:** When using auto-extraction from ``.README`` files, the extracted type mappings will be applied to the column specified by ``--sntype_var``. 

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
- The supernova/light-curve types supported can be changed using ``--sntypes``. If not provided, types are auto-detected from ``.README`` files in ``raw_dir`` or use built-in defaults (7 classes). If a type present in the data is not given as input in ``--sntypes``, it will be automatically assigned to a ``contaminant`` class (see below).
- The class used as target 0 (class of interest) for binary classification is controlled by ``--target_sntype`` (default: ``Ia``). This class is also placed first (index 0) in multiclass targets, ensuring consistency between binary and multiclass columns. If the specified ``--target_sntype`` value is not found in ``--sntypes``, the first entry in the dictionary is used as a fallback.

Automatic sntypes resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SuperNNova automatically determines supernova type mappings using the following priority order:

**1. Manual input (highest priority)**

Explicitly provide type mappings via ``--sntypes`` as a JSON dictionary:

.. code-block:: bash

    snn make_data --dump_dir <path> --raw_dir <path> --sntypes '{"101":"Ia", "120":"IIP", "132":"Ib"}'

This can also be specified in a YAML configuration file:

.. code-block:: yaml

    sntypes:
      "101": "Ia"
      "120": "IIP"
      "132": "Ib"

When ``--sntypes`` is provided, automatic detection is skipped and your specification is used.

**2. Auto-extraction from .README files**

If ``--sntypes`` is not provided, SuperNNova searches for ``*.README`` files in the ``raw_dir`` and extracts type mappings from the ``GENTYPE_TO_NAME`` block. This is the standard format used by SNANA simulations:

.. code-block:: none

    GENTYPE_TO_NAME:  # GENTYPE-integer (non)Ia transient-Name FITS-prefix
      1:   Ia       SALT3.             SNIaMODEL00
      20:  nonIa    SNIIP              NONIaMODEL03
      32:  nonIa    SNIb               NONIaMODEL01

For each GENTYPE number N found, two entries are created following the SNANA photo-ID convention:

- **N** → type name (e.g., ``"1": "Ia"``)
- **N+100** → type name (e.g., ``"101": "Ia"``)

For Ia types (column 2 == "Ia"), the type name is "Ia". For non-Ia types, the transient name from column 3 is used (e.g., "SNIIP", "SNIb").

**3. Built-in defaults (fallback)**

If no ``--sntypes`` is provided and no ``.README`` is found, SuperNNova uses built-in defaults:

.. code-block:: python

    {"101": "Ia", "120": "IIP", "121": "IIn", "122": "IIL1",
     "123": "IIL2", "132": "Ib", "133": "Ic"}

A message is printed indicating which method was used to determine the types.

Handling missing types (contaminant auto-detection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When building a database, the data may contain supernova types that are not listed in the ``--sntypes`` dictionary. Rather than failing or requiring you to manually enumerate every type in the data, SuperNNova automatically detects these missing types and assigns them to a ``contaminant`` class.

**How it works:**

- Before any class mapping, all unique type values in the data are compared against the keys in ``--sntypes``.
- Any type present in the data but missing from ``--sntypes`` is added to the dictionary with the value ``contaminant``.
- A yellow warning is printed listing the types that were auto-assigned.

**Effect on classification targets:**

- **Binary classification** (``nb_classes=2``): contaminants are treated as non-target (class 1), just like any other non-target type. No data is lost.
- **Multiclass classification** (``nb_classes>2``): contaminants are grouped into a single additional class. For example, if you specify 2 types (``Ib/c`` and ``Ia``) with ``--target_sntype Ia``, the multiclass target will have 3 classes: ``Ia`` (0), ``Ib/c`` (1), and ``contaminant`` (2).

**Example:**

Suppose your data contains types ``111, 112, 113, 115, 212`` but you only care about two:

.. code-block:: bash

    snn make_data --dump_dir <dump> --raw_dir <raw> --sntypes '{"112":"Ib/c", "113":"Ia"}'

SuperNNova will print:

.. code-block:: none

    [Missing sntypes] ['111', '115', '212'] assigned to 'contaminant' class

The resulting database will contain:

- ``target_2classes``: ``Ia`` (113) as class 0, everything else as class 1
- ``target_3classes``: ``Ia`` (113) as class 0, ``Ib/c`` (112) as class 1, ``contaminant`` (111, 115, 212) as class 2

This means you no longer need to manually list every type in the data when using ``--sntypes``. Only the types you want to distinguish need to be specified.

Handling unused types in sntypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``--sntypes`` contains types that are **not** present in the data (e.g. you specify ``"120":"IIP"`` but no type 120 exists in this dataset), those entries are kept in the dictionary and a warning is printed:

.. code-block:: none

    [Unused sntypes] Keys ['120'] not found in data (kept for class structure consistency)

The unused types are **not** removed because preserving the full ``--sntypes`` dictionary ensures that the ``target_Nclasses`` column name and class indices remain stable across datasets. This is important when you train a model on one dataset and classify another that may contain a different subset of types — the class structure must match for the model predictions to be meaningful.

The balanced downsampling used during training is unaffected because ``groupby`` only operates on classes that have data; empty classes are simply absent from the groups.

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