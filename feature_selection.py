# from nearest_neighbour import leave_one_out_accuracy

# def forward_selection(data):
#     m = data.shape[1] - 1
#     current_set = []
#     results = []
#     for _ in range(m):
#         best_feature = None
#         best_acc = 0.0
#         for f in range(1, m + 1):
#             if f not in current_set:
#                 trial = current_set + [f]
#                 acc = leave_one_out_accuracy(data, trial)
#                 if acc > best_acc:
#                     best_acc = acc
#                     best_feature = f
#         if best_feature is None:
#             break
#         current_set.append(best_feature)
#         results.append((list(current_set), best_acc))
#     return results

# def backward_elimination(data):
#     m = data.shape[1] - 1
#     current_set = list(range(1, m + 1))
#     results = []
#     while len(current_set) > 1:
#         best_subset = None
#         best_acc = 0.0
#         for f in current_set:
#             trial = [x for x in current_set if x != f]
#             acc = leave_one_out_accuracy(data, trial)
#             if acc >= best_acc:
#                 best_acc = acc
#                 best_subset = trial
#         if best_subset is None:
#             break
#         current_set = best_subset
#         results.append((list(current_set), best_acc))
#     return results


from nearest_neighbour import compute_loocv_accuracy

def forward_feature_selection(data_matrix):
  n_features = data_matrix.shape[1] - 1
  selected_features = []
  selection_results = []
  for _ in range(n_features):
    best_feature = None
    best_accuracy = 0.0
    for f in range(1, n_features + 1):
      if f not in selected_features:
        accuracy = compute_loocv_accuracy(data_matrix, selected_features + [f])
        if accuracy > best_accuracy:
          best_accuracy = accuracy
          best_feature = f
    if best_feature is None:
      break
    selected_features.append(best_feature)
    selection_results.append((selected_features.copy(), best_accuracy))
  return selection_results

def backward_feature_elimination(data_matrix):
  n_features = data_matrix.shape[1] - 1
  selected_features = list(range(1, n_features + 1))
  elimination_results = []
  while len(selected_features) > 1:
    best_subset = None
    best_accuracy = 0.0
    for f in selected_features:
      trial = [x for x in selected_features if x != f]
      accuracy = compute_loocv_accuracy(data_matrix, trial)
      if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_subset = trial
    if best_subset is None:
      break
    selected_features = best_subset
    elimination_results.append((selected_features.copy(), best_accuracy))
  return elimination_results
