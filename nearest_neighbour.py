


import numpy as np

def load_dataset(path):
  data_matrix = []
  with open(path, 'r') as f:
    for line in f:
      tokens = line.strip().split()
      if tokens:
        row = [float(x) for x in tokens]
        data_matrix.append(row)
  return np.array(data_matrix)

def compute_loocv_accuracy(data_matrix, feature_indices):
  n_rows = data_matrix.shape[0]
  correct = 0
  for i in range(n_rows):
    test_row = data_matrix[i]
    train_matrix = np.delete(data_matrix, i, axis=0)
    if feature_indices:
      test_values = test_row[feature_indices]
      train_values = train_matrix[:, feature_indices]
    else:
      test_values = np.array([])
      train_values = np.empty((train_matrix.shape[0], 0))
    differences = train_values - test_values
    distances = np.linalg.norm(differences, axis=1)
    nearest = np.argmin(distances)
    if train_matrix[nearest, 0] == test_row[0]:
      correct += 1
  return correct / n_rows
