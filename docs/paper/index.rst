
Paper reproduction walkthrough
================================

Reproducing the models
----------------------
To reproduce the stats on the paper you need first to run all the models

.. code::

    python run_paper.py

With a GPU and a ``--batch_size = 128`` (default) this takes around two weeks. If you increase ``batch_size`` it may be reduced to a couple of days but performance can be slightly reduced.


Reproducing the stats and plots
---------------------------------

Summary statistics for all trained models and a printout with the stats and plots used in the paper are produced by:

.. code::

    python run.py --performance

Summary statistics will be found in ``snndump/stats/summary_stats.csv``. Statistics used in the paper are printed out and latex tables created in ``snndump/latex/``. Plots and figures are found in ``snndump/figures/`` and ``snndump/lightcurves/``.

To obtain summary statistics only, comment in the two lines after ``# Stats and plots in paper`` in ``run.py``.

To obtain stats only, comment the plotting function at ``SuperNNova/supernnova/paper/paper_thread.py`` by changing ``SuperNNova_stats_and_plots_thread(df_stats, settings, plots=False)``.



