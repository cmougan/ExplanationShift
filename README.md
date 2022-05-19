# Explanation Shift
## Detecting and quantifying the decay of predictive performance and fairness on tabular data
**Abstract**
As input data distributions evolve, the predictive performance of machine learning models tends to deteriorate. In the past, predictive performance was considered the key indicator to monitor. However, other indicators have come to attention within the last years, such as fairness and explanation aspects. In this work, we investigate how predictive model performance, model explanation characteristics, and model fairness are affected under distribution shifts and how these key indicators are related to each other for tabular data.
We find that the modeling of explanation shifts can be a better indicator for the decay of predictive performance and fairness than state-of-the-art techniques based on representations of distribution shifts. We provide a mathematical analysis of synthetic examples and experimental evaluation of real-world data.

## Experiments
The experimental section is divided into two main parts. Experiments with synthetic data and experiments using the folks datasets.

### Synthetic Data Experiments

- Detecting multivariate shift `gaussianShift.py`
- Posterior distribution shift `posteriorShift.py`
- Quantifying model degradation `distributionSamplingGaussian.py`

### Experiments on folks dataset
- `samplingOOD.py`

Inside the results folder the scripts that recompile the output of the above experiment and produce the visualizations used in the paper.