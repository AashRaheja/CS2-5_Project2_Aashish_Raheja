

from nearest_neighbour import load_dataset, compute_loocv_accuracy

small_dataset = load_dataset("/Users/aashishraheja/Desktop/Aashish/Spring_2025/CS205_Feature_Selection/CS205_small_Data__26.txt")
large_dataset = load_dataset("/Users/aashishraheja/Desktop/Aashish/Spring_2025/CS205_Feature_Selection/CS205_large_Data__14.txt")

best_small_accuracy = compute_loocv_accuracy(small_dataset, [11, 6, 10])
best_large_accuracy = compute_loocv_accuracy(large_dataset, [25, 8])

print("Final small-set accuracy (features 11, 6, 10):", best_small_accuracy)
print("Final large-set accuracy (features 25, 8):", best_large_accuracy)
