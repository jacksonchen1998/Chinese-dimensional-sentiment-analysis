# Chinese-dimensional-sentiment-analysis
2024 NYCU Natural Language Processing

## Dataset

- Training Dataset: Chinese EmoBank (CVAT)
- Testing Dataset: 1000+ Mental Health Texts

## Method

This method aims to leverage a set of six independently trained BERT-based models, each specializing in a different category. 

The final prediction is derived from the average of all model outputs, providing a robust and reliable estimate.

## Evluation

Mean Absolute Error (MAE):

$$
  MAE = \frac{1}{n} \sum_{i=1}^n |a_i - p_i|
$$

Person Correlation Coefficient ($r$):

$$
  r = \frac{1}{n-1} \sum_{i}^n (\frac{a_i - \mu_A}{\sigma_A})(\frac{p_i - \mu_P}{\sigma_A})
$$

- $a_i \in A$: $i$-th actual value
- $p_i \in P$: $i$-th predicted value
- $\mu$: mean value
- $\sigma$: standard deviation
- $n$: the number of test sample

A lower MAE and a higher r indicate more accurate prediction performance.

## Reference

- [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)
- [Chinese EmoBank: Building Valence-Arousal Resources for Dimensional Sentiment Analysis](https://dl.acm.org/doi/full/10.1145/3489141)


















