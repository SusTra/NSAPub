# NSAPub
Neighbourhood Sustainability Assessment using Publicly available data

## Main files
 * [`01_traffic_google_preprocess.ipynb`](01_traffic_google_preprocess.ipynb): data validation, preprocessing and export for further analyses of travel time data.
 * [`02_traffic_google_basic.ipynb`](02_traffic_google_basic.ipynb): data import, validation, basic visualisation and some tests of travel time data.
  * [`03_traffic_google_analyse_circadian.ipynb`](03_traffic_google_analyse_circadian.ipynb): circadian analyis  of travel time data using the CosinorPy package (located in the `CosinorPy` folder). Results are exported to `cosinor_results*` folders.
  * [`04_traffic_counters_preprocess.ipynb`](04_traffic_counters_preprocess.ipynb): merge and preprocess the counter data.
  * [`05_traffic_counters_basic.ipynb`](05_traffic_counters_basic.ipynb): basic analysis and validation of counter data.
  * [`06_traffic_google_add_counters.ipynb`](06_traffic_google_add_counters.ipynb): merge travel time data with counter data.
  * [`07_pace_regression.ipynb`](07_pace_regression.ipynb): establish, evaluate and plot the regression functions for the assessment of travel time data using counter data.
  
## Folders
 * `data`: input data and data produced by preprocessing and used by further analyses.
 * `cosinor_results*`: result of the cosinor analysis.
 * `regression_results*`: result of the regression analysis.
 * `figs`: selected figures.

 





