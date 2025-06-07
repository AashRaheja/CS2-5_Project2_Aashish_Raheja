import numpy as np
import matplotlib.pyplot as plt
from nearest_neighbour import load_dataset
from feature_selection import forward_feature_selection, backward_feature_elimination

def plot_and_save(data_path, selection_fn, title, filename):
  data = load_dataset(data_path)
  results = selection_fn(data)
  counts = [len(fset) for fset, _ in results]
  accs = [acc for _, acc in results]
  plt.figure()
  plt.plot(counts, accs, marker='o', linewidth=1.5)
  plt.xlabel("Number of Features")
  plt.ylabel("LOOCV Accuracy")
  plt.title(title)
  plt.xticks(counts)
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.tight_layout()
  plt.savefig(filename)
  plt.close()

def main():
  small_path = "/Users/aashishraheja/Desktop/Aashish/Spring_2025/CS205_Feature_Selection/CS205_small_Data__26.txt"
  large_path = "/Users/aashishraheja/Desktop/Aashish/Spring_2025/CS205_Feature_Selection/CS205_large_Data__14.txt"

  plot_and_save(small_path, forward_feature_selection,
    "Figure 1: Forward Selection on Small Dataset",
    "figure1_forward_small.png")

  plot_and_save(small_path, backward_feature_elimination,
    "Figure 2: Backward Elimination on Small Dataset",
    "figure2_backward_small.png")

  plot_and_save(large_path, forward_feature_selection,
    "Figure 3: Forward Selection on Large Dataset",
    "figure3_forward_large.png")

  plot_and_save(large_path, backward_feature_elimination,
    "Figure 4: Backward Elimination on Large Dataset",
    "figure4_backward_large.png")

  print("Saved: figure1_forward_small.png, figure2_backward_small.png, figure3_forward_large.png, figure4_backward_large.png")

if __name__ == "__main__":
  main()
