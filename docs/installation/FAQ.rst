
FAQ
=====================

General questions
--------------------

- **What is SuperNNova?**

SuperNNova is a framework for lightcurve classification which uses supervised learning algorithms. Training of these algorithms rely on large annotated databases. Typically, we use simulations as the training set.

- **Do you have a paper describing SuperNNova? How can I cite you?**

We just submitted the paper to the journal. But a copy of the paper can be found here `ArXiv`_.

- **What data do I need?**

You only need lightcurves (photometric time-series) to use SuperNNova. Additional information can be added as well. E.g. we used supernova host-galaxy redshifts in the paper.

- **Is the data used in the paper publicly available?**

Yes it is! `SuperNNovaSimulations`_
We want to foster reproducibility so you can copy the data and reproduce all our experiments with ``run_paper.py``. Beware, it will take while!

- **How did you create the simulations used in the paper?**

We used SNANA to generate the supernovae lightcurves. Our data is similar to the Supernova Photometric Classification Challenge (SPCC) data with updated models used in the DES simulations.

- **Why use SuperNNova?**

First, it is open source, so you can modify it for your science goal or just see for yourself what is the "blackbox". Second, we have pretty good performance. Third, we also provide Bayesian interpretations of RNN which allow better uncertainty handling, which is useful for cosmological analyses.

- **Can I use SuperNNova for my classification problem?**

Please do but beware: you need to have a large amount of lightcurves (simulated or data) per type of event you are trying to classify, otherwise performance is pretty poor.

- **How can I use SuperNNova for my classification problem?**

It may require a little bit of code modification depending on your data. We can load data from SNANA formats (``.FITS`` and ``FITRES``, the latter is an ascii file) or ``.csv`` files (like the one from the Kaggle challenge, PlastiCC). Observations are grouped per night, so if you are looking for fast transients, you may need to create your own data pipeline. Contact us if you have questions!


Technical questions
--------------------

- **What algorithms are available for classification?**

Currently we have a Baseline RNN and two Bayesian RNNs. The Bayesian RNNs are based on the work of `Fortunato et al 2017`_ and `Gal et Ghahramani 2015`_ and allow us to estimate prediction uncertainty. These algorithms require only raw lightcurve data. We have also a Random Forest classifier that relies lightcurve features. You can obtain these with fitters: an exponential that rises and falls or a type Ia supernova `SALT2`_ fits.

- **Why is training slow ?**

If you have a GPU, you can activate training on GPU with the ``--use_cuda`` flag.
Alternatively, you may select a smaller data fraction ``--data_fraction 0.1`` to train on a smaller set.


- **OSError: Unable to open file (unable to open file: name = '/home/snndump/processed/DES_database.h5'**

You have probably forgotten to set your ``dump_dir`` correctly. Provide the ``--dump_dir`` argument correctly

- **Where do I find the model naming scheme?**

You can find it in ``SuperNNova/utils/ExperimentSettings.py`` under ``model_name``. A start guide can be found in our :ref:`Start`.

- **How do I change the directory where the data can be found?**

You can give add to your terminal command ``--dump_dir foldername``. This folder should have the same structure as our data repositroeis (see :ref:`Data`).

- **If I trained several models, is there a way to see a summary of the statistics?**

Yes, you need to call ``python run.py --performance``. It will be created in ``{dump_dir}/stats`` as ``summary_stats.csv``. It will compute various metrics which can be averaged over multiple random seeds.



.. _ArXiv: https:/arxiv.org
.. _SuperNNovaSimulations: http://mso.anu.edu.au/~anais/SuperNNova.html
.. _Fortunato et al 2017: https://arxiv.org/abs/1704.02798
.. _Gal et Ghahramani 2015: https://arxiv.org/abs/1506.02142
.. _SALT2: https://arxiv.org/pdf/astro-ph/0701828.pdf
