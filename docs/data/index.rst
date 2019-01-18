.. _DataStructure:

Data walkthrough
=========================

Recommended code organization structure:

.. code::

    ├── sndump        (to save the data)
    │   ├── raw
    │
    ├── supernnova
    │   ├── supernnova
    │   ├── docker
    │   ├── docs
    │   ├── tests


**To build the database:**

- Ensure you have raw data saved to ``{dump_dir}/raw``
- The default settings assume the raw data is saved to ``sndump/raw``
- You can save the data in any folder, but you then have to specify the ``dump_dir`` with the ``--dump_dir XXX`` command.


Creating a debugging database
-------------------------------

.. code::

    python run.py --data --dump_dir tests/dump

- You do not need to download the raw data for this database, it is already included in ``tests/dump``
- This creates a database for a very small subset of all available data
- This is intended for debugging purposes (training, validation can run very fast with this small database)
- The database is saved to the specified ``tests/dump/processed``


Creating a training database
------------------------------

.. code::

    python run.py --data --dump_dir <path/to/full/database/>

- You **DO NEED** to download the raw data for this database
- This creates a database for all of the available data
- The database is saved to the specified ``dump_dir``, in the ``processed`` subfolder


Under the hood
-------------------------------

Preparing data splits
~~~~~~~~~~~~~~~~~~~~~~

We first compute the data splits:

- The HEAD FITS files are analyzed to compute 80/10/10 train/test/val splits.
- The splits are different for the salt/photometry datasets
- The splits are different depending on the classification target
- We downsample the dataset so that for a given classification task, all classes have the same cardinality

Preprocessing
~~~~~~~~~~~~~~

We then pre-process each FITS file

- Join column from header files
- Select columns that will be useful later on
- Compute SNID to tag each light curve
- Compute delta times between measures
- Removal of delimiter rows


Pivot
~~~~~~~~~~~~~~

We then pivot each FITS file: we will group time-wise close observations on the same row
and each row in the dataframe will show a value for each of the flux and flux error column

- All observations withing 8 hours of each other are assigned the same MJD
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
    ├── dataset_photometry_3classes     (0: train set, 1: valid set, 2: test set, -1: not used)
    ├── dataset_photometry_7classes     (0: train set, 1: valid set, 2: test set, -1: not used)
    │
    ├── target_photometry_2classes      (integer between 0 and 1, included)
    ├── target_photometry_3classes      (integer between 0 and 2, included)
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