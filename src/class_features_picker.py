from pandas import Series


class TopFeaturesPicker:
    def __init__(self) -> None:
        pass

    def pick_top_features(self, feature_importances: Series, percentage: float = 0.95):
        sorted_importances = feature_importances.sort_values(ascending=False)
        total_importance = sorted_importances.sum()
        importance_threshold = total_importance*percentage
        cumulative_sum = sorted_importances.cumsum()
        # find the first index that exceeds the threshold:
        cutoff_index = (cumulative_sum >= importance_threshold).to_numpy().argmax()

        selected_features = sorted_importances.iloc[:cutoff_index + 1].index.to_list()
        return selected_features

        
