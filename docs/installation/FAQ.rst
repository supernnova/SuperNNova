
FAQ
=====================

General questions
--------------------

- **What is SuperNNova?**

SuperNNova is a framework for lightcurve classification which uses supervised learning algorithms. Training of these algorithms rely on large annotated databases. Typically, we use simulations as the training set.

- **Do you have a paper describing SuperNNova? How can I cite you?**

The SuperNNova paper has been published by `MNRAS`_. A copy of the paper can be found here `ArXiv`_. 

- **How can I install it?**

You can ``clone`` our `GitHub`_. Beware, the supported version of GitHub repository is this `GitHub`_!!!! (previous version was hosted in a different webpage). If you have a compelling case to bring it back let me know! SuperNNova works in Unix based systems only.

- **I have installed SuperNNova previously and I can't find the previous documentation**

At the end of 2024 we have released a new version of SuperNNova with updated pytorch libraries and a new structure. The previous code can be found in the `GitHub`_ repository under the branch ``SNANA_DES5yr``. To access the documentation `_SNANADes5yr docs_`.

- **What data do I need?**

You only need lightcurves (photometric time-series) to use SuperNNova. Additional information can be added as well. E.g. we used supernova host-galaxy redshifts. If you use redshifts and want to do a cosmology analysis you should also use the norm `cosmo_quantile` to avoid biases (see the paper `5-year photometric sample`)

- **Is the data used in the 2020 paper publicly available?**

Yes it is! `SuperNNovaSimulations`_
We want to foster reproducibility so you can copy the data and reproduce all our experiments with ``run_paper.py`` in the ``paper`` branch. Beware, it will take while!

- **How did you create the simulations used in the method paper?**

We used `SNANA`_ to generate the supernovae lightcurves. Our data is similar to the Supernova Photometric Classification Challenge (SPCC) data with updated models used in the DES simulations.

- **Why use SuperNNova?**

First, it is open source, so you can modify it for your science goal or just see for yourself what is the "blackbox". Second, we have pretty good performance. Third, we also provide Bayesian interpretations of RNN which allow better uncertainty handling, which is useful for cosmological or any statistical analyses.

- **Can I use SuperNNova for my classification problem?**

Please do! But beware: you need to have a large amount of lightcurves (simulated or data) per type of event you are trying to classify, otherwise performance is pretty poor.

- **How can I use SuperNNova for my classification problem?**

It may require a little bit of code modification depending on your data. You can load data from SNANA formats (``.FITS`` and ``FITRES``, the latter is an ascii file) or ``.csv`` files (like the one from the Kaggle challenge, PlastiCC). Observations are grouped per night, so if you are looking for fast transients, you may need to create your own data pipeline or modify SuperNNova time grouping. Contact us if you have questions amoller@swin.edu.au and please report any issues!

- **Has SuperNNova been used for scientific analyses?**

Yes! It has been used both for supernova classification in the Dark Energy Survey `5-year photometric sample`_ and `Fink broker`_ with ZTF data. More analyses with different transients in progress!

Technical questions
--------------------

- **What algorithms are available for classification?**

Currently we have a Baseline RNN and two Bayesian RNNs. The Bayesian RNNs are based on the work of `Fortunato et al 2017`_ and `Gal et Ghahramani 2015`_ and allow us to estimate prediction uncertainty. These algorithms require only raw lightcurve data.

- **Why is training slow ?**

If you have a GPU, you can activate training on GPU with the ``--use_cuda`` flag.
Alternatively, you may select a smaller data fraction ``--data_fraction 0.1`` to train on a smaller set.

- **Where do I find the model naming scheme?**

You can find it in ``python/supernnova/utils/experiment_settings.py`` under ``model_name``. A start guide can be found in our :ref:`Start`.

- **How do I change the directory where the data can be found?**

You can give add to your terminal command ``--dump_dir foldername``. This folder should have the same structure as our data repositories (see :ref:`Data`).

- **If I trained several models, is there a way to see a summary of the statistics?**

Yes, you need to call ``snn performance``. It will be created in ``{dump_dir}/stats`` as ``summary_stats.csv``. It will compute various metrics which can be averaged over multiple random seeds.


Common issues
--------------------
- **OSError: Unable to open file (unable to open file: name = '/home/snndump/processed/DES_database.h5'**

You have probably forgotten to set your ``dump_dir`` correctly. Provide the ``--dump_dir`` argument correctly.

- **ValueError: No objects to concatenate in Database creation**
Check that you provided the appropriate ``raw_dir`` and that the files are either .FITS tables or .csv with the proper format (see data tab). Another possibility is that you are using a different survey than DES and you need to specify the appropriate ``list_filters``.

- **ValueError during training when using a custom** ``--sntypes`` **with only a subset of types**

Previously, if your data contained types not listed in ``--sntypes``, training would fail with a ``ValueError`` because the missing types were assigned to an out-of-bounds class index. This has been fixed: missing types are now automatically assigned to a ``contaminant`` class. You only need to specify the types you want to distinguish in ``--sntypes``. See :ref:`DataStructure` for details on how contaminant auto-detection works.

- **Your error not here? Please open an issue in GitHub! **


.. _ArXiv: https://arxiv.org/abs/1901.06384
.. _MNRAS: https://academic.oup.com/mnras/advance-article-abstract/doi/10.1093/mnras/stz3312/5651173
.. _SNANADes5yr docs: https://supernnova.readthedocs.io/snana_des5yr/index.html
.. _SuperNNovaSimulations: https://zenodo.org/record/3265189#.XRo2mS2B1TY
.. _Fortunato et al 2017: https://arxiv.org/abs/1704.02798
.. _Gal et Ghahramani 2015: https://arxiv.org/abs/1506.02142
.. _SNANA: https://arxiv.org/abs/0908.4280
.. _GitHub: https://github.com/supernnova/SuperNNova
.. _5-year photometric sample: https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.5159M/abstract
.. _Fink broker: https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.3272M/abstract 
