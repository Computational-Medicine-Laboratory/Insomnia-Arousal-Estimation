# Insomnia Arousal Estimation

This code is written by Tianyu Hwang in affiliation with Computational Medicine Lab at New York University. Tianyu was supervised by Rose T. Faghih during this project. Please contact Rose T. Faghih (email: rtfaghih@nyu.edu) for any question regarding the code.

Please refer to the following paper if you use any part of the code:

[1] Hwang et al. Disrupted fear regulation in insomnia disorder revealed by arousal estimation from skin conductance.

## Dataset

The dataset (`dataset.csv`) used in this repository is provided as Supporting Information (S1 Data) in the paper [1]. It contains preprocessed signals sampled at 4 Hz in long format.

## Installation

To install the required packages, execute:

```bash
pip install -r requirements.txt
```

## Running the Code

To run the estimation code, execute:

```bash
python main.py
```

To plot the results, execute:

```bash
python plot.py
```

To run statistical analysis, execute:

```bash
python stats.py
```

## Results

The results will be saved in the `results` directory.
