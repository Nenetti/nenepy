from cleanlab.pruning import get_noise_indices

ordered_label_errors = get_noise_indices(
    s=numpy_array_of_noisy_labels,
    psx=numpy_array_of_predicted_probabilities,
    sorted_index_method='normalized_margin',  # Orders label errors
)

from cleanlab.classification import LearningWithNoisyLabels
from sklearn.linear_model import LogisticRegression

# Wrap around any classifier. Yup, you can use sklearn/pyTorch/Tensorflow/FastText/etc.
