The CellSort script `cellsort.py` uses Python 2.7. Required packages include `FlowCytometryTools` and `scikit-learn`.

Make sure to check the `scikit-learn` version -- the newer versions may have incompatible changes in API. Version `0.15.0` works well. If you have a newer version, install the older version (e.g. `pip install scikit-learn==0.15.0`) and add the following to the top of the script before the `sklearn` import statements:

```python
import pkg_resources
pkg_resources.require("scikit-learn==0.15.0")
```

To run the algorithm, use:

```python
python cellsort.py NEGATIVE_CONTROL POSITIVE_CONTROL
```

where `NEGATIVE_CONTROL` and `POSITIVE_CONTROL` are `.fcs` files containing negative and positive control data.

By default, the script uses the channels for `FITC-A` and `PE-Texas Red-A`. The provided sample negative and positive data use `Alexa Fluor 488-A` instead of `FITC-A`, which can be provided to the script using the `-L` flag:

```python
python cellsort.py neg_data.fcs pos_data.fcs -L 'Alexa Fluor 488-A' 'PE-Texas Red-A'
```

Include the `-g` flag to display a plot of the data with gate.

Additional options can be listed using `python cellSVCRn.py -h`.
