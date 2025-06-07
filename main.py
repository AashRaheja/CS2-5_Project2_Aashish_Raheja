


from nearest_neighbour import load_dataset
from feature_selection import forward_feature_selection, backward_feature_elimination

def main():
  small_path = "/Users/aashishraheja/Desktop/Aashish/Spring_2025/CS205_Feature_Selection/CS205_small_Data__26.txt"
  large_path = "/Users/aashishraheja/Desktop/Aashish/Spring_2025/CS205_Feature_Selection/CS205_large_Data__14.txt"
  small_dataset = load_dataset(small_path)
  large_dataset = load_dataset(large_path)

  print("Forward Selection on small dataset:")
  for feature_set, accuracy in forward_feature_selection(small_dataset):
    print("Features:", feature_set, "Accuracy:", accuracy)

  print()
  print("Backward Elimination on small dataset:")
  for feature_set, accuracy in backward_feature_elimination(small_dataset):
    print("Features:", feature_set, "Accuracy:", accuracy)

  print()
  print("Forward Selection on large dataset:")
  for feature_set, accuracy in forward_feature_selection(large_dataset):
    print("Features:", feature_set, "Accuracy:", accuracy)

  print()
  print("Backward Elimination on large dataset:")
  for feature_set, accuracy in backward_feature_elimination(large_dataset):
    print("Features:", feature_set, "Accuracy:", accuracy)

if __name__ == "__main__":
  main()
