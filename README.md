# NSAPub: Neighbourhood Sustainability Assessment using Publicly available data

This repository is accompanying the paper **Integrative approach for neighbourhood sustainability assessment using publicly available traffic data** by *Š. Verovšek, M.  Juvančič, S. Petrovčič, T. Zupančič, M. Svetina, M. Janež, Ž. Pušnik, N. Velikajne, and M. Moškon*.


## Main files
 * [`traffic_google_preprocess.ipynb`](traffic_google_preprocess.ipynb): data validation, preprocessing and export for further analyses of travel time data.
 * [`traffic_google_basic.ipynb`](traffic_google_basic.ipynb): data import, validation, basic visualisation and some tests of travel time data.
  * [`traffic_google_analyse_circadian.ipynb`](traffic_google_analyse_circadian.ipynb): circadian analyis  of travel time data using the CosinorPy package (located in the `CosinorPy` folder). Results are exported to `cosinor_results*` folders.
  * [`traffic_counters_preprocess.ipynb`](traffic_counters_preprocess.ipynb): merge and preprocess the counter data.
  * [`traffic_counters_basic.ipynb`](traffic_counters_basic.ipynb): basic analysis and validation of counter data.
  * [`traffic_google_add_counters.ipynb`](traffic_google_add_counters.ipynb): merge travel time data with counter data.
  * [`pace_regression.ipynb`](pace_regression.ipynb): establish, evaluate and plot the regression functions for the assessment of travel time data using counter data.
  * [`reliability_performance_measures.ipynb`](reliability_performance_measures.ipynb): plot the reliability performance measures graph on a selected route.
  * [`partial_correlation_and_travel_indices.ipynb`](partial_correlation_and_travel_indices.ipynb): calculate partial correlations between the observed variables; evaluate reliability performance measures, namely, travel time index (TTI) and planning time index (PTI).
  
## Folders
 * `data`: input data and data produced by preprocessing and used by further analyses.
 * `cosinor_results*`: result of the cosinor analysis.
 * `regression_results*`: result of the regression analysis.
 * `figs`: selected figures.

 