import numpy as np

from metrics.weighted_mae import WeightedMAE

HR_GRID = list(range(30, 230, 1))


def test_weighted_mae():
    y_true = np.array(HR_GRID)
    y_pred = np.array(HR_GRID)
    m = WeightedMAE()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == 0.0


def test_weighted_mae_batch():
    y_true = np.random.random(0, 1, size=(32, 200))
    y_pred = y_true.copy()
    m = WeightedMAE()
    m.update_state(y_true, y_pred)
    assert m.result().numpy() == 0.0
