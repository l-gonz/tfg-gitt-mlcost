# Machine learning model cost comparison dashboard

This python script runs a comparison of several machine learning models on a dataset and outputs the accuracy of the model, the time employed for training and the CO2 emissions generated by the process. CO2 emissions are calculated using [CodeCarbon](https://github.com/mlco2/codecarbon).

An example of the measures obtained for the included datasets can be seen on the [output](output.txt) file.

## Setup
1. Clone repository
2. Create virtual environment for managing dependencies
```
conda create -n mlcost -f environment.yml
```
3. Activate the environment
```
conda activate mlcost
```

## Execution
Run the program without any options to use the default Iris dataset.

There are several larger datasets available for testing in the [data](data) folder:

- To run the program with a specific dataset, use the `-d` option.
- To specify a separate file with the test data, use the `-t` option.
- If the dataset uses a different format from a standard comma-separated values, use the `-s` to specify the separator.

### Examples
#### Adult dataset
```
python src/main.py -d data/adult/adult.data -t data/adult/adult.test -s ", "
```

#### Hepatitis dataset
```
python src/main.py -d data/hepatitis/hepatitis.data
```