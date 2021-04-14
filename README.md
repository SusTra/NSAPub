# NSAPub
Neighbourhood Sustainability Assessment using Publicly available data

## Files
 * [`01_traffic_google_preprocess.ipynb`](https://github.com/mmoskon/SusTra/blob/mmoskon/01_traffic_google_preprocess.ipynb): data validation, preprocessing and export for further analyses of travel time data.
 * [`02_traffic_google_basic.ipynb`](https://github.com/mmoskon/SusTra/blob/mmoskon/02_traffic_google_basic.ipynb): data import, validation, basic visualisation and some tests of travel time data.
  * [`03_traffic_google_analyse_circadian.ipynb`](https://github.com/mmoskon/SusTra/blob/mmoskon/03_traffic_google_analyse_circadian.ipynb): circadian analyis  of travel time data using the CosinorPy package (located in the `CosinorPy` folder). Results are exported to `cosinor_results*` folders.
  * [`04_traffic_counters_preprocess.ipynb`](https://github.com/mmoskon/SusTra/blob/mmoskon/04_traffic_counters_preprocess.ipynb): merge and preprocess the counter data.
  * [`05_traffic_counters_basic.ipynb`](https://github.com/mmoskon/SusTra/blob/mmoskon/05_traffic_counters_basic.ipynb): basic analysis and validation of counter data.
  * [`06_traffic_google_add_counters.ipynb`](https://github.com/mmoskon/SusTra/blob/mmoskon/06_traffic_google_add_counters.ipynb): merge travel time data with counter data.
  * [`07_pace_regression.ipynb`](https://github.com/mmoskon/SusTra/blob/mmoskon/07_pace_regression.ipynb): establish, evaluate and plot the regression functions for the assessment of travel time data using counter data.
  
## Folders
 * `data`: input data and data produced by preprocessing and used by further analyses.
 * `cosinor_results*`: result of the cosinor analysis.
 * `regression_results*`: result of the regression analysis.
 * `figs`: selected figures.

## TODO
 * regresija na večji skali
 

## Analysis procedure (counters)
1. List the counters you want to use in [`data/stevci/counter_ids.txt`](https://github.com/mmoskon/SusTra/blob/mmoskon/data/stevci/counter_ids.txt) (values separated by `;`).
2. Run [`data/stevci/copy_files.py`](https://github.com/mmoskon/SusTra/blob/mmoskon/data/stevci/copy_files.py) to copy the counter data into a folder that will be used for the analysis.
3. Define the contents of [`data/counters_per_route.txt`](https://github.com/mmoskon/SusTra/blob/mmoskon/data/counters_per_route.txt) (values separated by `;`, first value in a row presents route ID, other values counter IDS - in the latter, postfix `-1` means in a reverse direction).
4. Define the contents of [`data/counter_synonyms.txt`](https://github.com/mmoskon/SusTra/blob/mmoskon/data/counter_synonyms.txt) to unify the counter IDs - sometimes, different IDs are used for the same counter (values separated by `;`, first value in a row presents the main ID (used as well in `counters_per_route.txt`), other values present alternative IDs).
5. Run `ipynb` files with prefixes `04` - `07`.




