

import numpy as np
import matplotlib.pyplot as plt
from feature_selection import forward_feature_selection

def load_vertebral_dataset(path):
  data_started = False
  parsed = []
  with open(path, 'r') as f:
    for raw_line in f:
      line = raw_line.strip()
      if not data_started:
        if line.lower() == "@data":
          data_started = True
        continue
      if not line or line.startswith('%'):
        continue
      tokens = [tok.strip() for tok in line.split(',')]
      if len(tokens) != 7:
        continue
      try:
        features = [float(tokens[i]) for i in range(6)]
      except ValueError:
        continue
      label_str = tokens[6]
      if label_str == 'Abnormal':
        label = 0
      elif label_str == 'Normal':
        label = 1
      else:
        continue
      parsed.append([label] + features)
  if not parsed:
    raise RuntimeError(f"No valid data loaded from {path}")
  return np.array(parsed, dtype=float)

def normalize_dataset(matrix):
  means = np.mean(matrix, axis=0)
  stds = np.std(matrix, axis=0)
  stds[stds == 0] = 1.0
  return (matrix - means) / stds

def main():
  data_path = "/Users/aashishraheja/Desktop/Aashish/Spring_2025/CS205_Feature_Selection/Part_2/column_2C_weka.arff"
  labeled_data = load_vertebral_dataset(data_path)
  labels = labeled_data[:, 0].astype(int)
  raw_features = labeled_data[:, 1:]
  normalized_features = normalize_dataset(raw_features)
  n_rows, _ = normalized_features.shape
  combined_data = np.hstack((labels.reshape(n_rows, 1), normalized_features))
  selection_results = forward_feature_selection(combined_data)
  feature_counts = [len(fset) for fset, _ in selection_results]
  accuracies = [acc for _, acc in selection_results]
  plt.figure(figsize=(7, 5))
  plt.plot(feature_counts, accuracies, marker='o', linewidth=1.5)
  plt.xlabel("Number of Features Selected")
  plt.ylabel("Leave-One-Out Accuracy")
  plt.title("Forward Selection on Vertebral Column (ARFF)")
  plt.xticks(feature_counts)
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.tight_layout()
  plt.show()
  print("Forward-Selection Order (feature indices refer to columns 1..6):")
  for feature_set, accuracy in selection_results:
    print("Features =", feature_set, "Accuracy =", f"{accuracy:.4f}")
  best_set, best_acc = max(selection_results, key=lambda x: x[1])
  print()
  print("Best feature subset:", best_set, f"(LOOCV accuracy = {best_acc:.4f})")

if __name__ == "__main__":
  main()
