from metrics.weighted_mae import WeightedMAE

HR_GRID = list(range(30, 230, 1))


class TwoLabelWeightedMAE(WeightedMAE):
    def __init__(self, name='two_label_weighted_mae', **kwargs):
        super(TwoLabelWeightedMAE, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        print(y_true)
        print(y_pred)
        return super().update_state(y_true[:, 0, :], y_pred, sample_weight=sample_weight)
