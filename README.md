# fBSM-calibration

Exploring and comparing calibration techniques of fractional Black Scholes Model

1. Python scripts in the Code directory modularise various steps in training and modelling.
> PricerWithNoPenalty, PricerWithPenalty and InverseMap are the gateway scripts <br>
> Other scripts are utility functions for various steps involved 

2. Experiments folder houses 2 classes ipython notebooks using components formally explained in Code directory
> Calibration.ipynb shows the experiments done for Pricers, Inverse Map and Inference<br>
> Hurst exponent-{stock}.ipynb shows the baseline estimations and market data analysis for model parameters corresponding to the particular stock

3. Pictures directory shows the various images pertaining to implementation used in the Project reports.

4. TrainingLogs directory has the logs of training and validation errors in each epochs. It also has the individual model pickle files.

<br>
<br>

*Note: All the experiments were done in Google Colab due to better hardware support for training.The 3rd party libraries used are PyTorch, Numpy, Scipy, Scikit, Matplotlib and Pandas, all of which are available in Colab by default.*