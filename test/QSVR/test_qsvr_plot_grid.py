import numpy as np

from pyqpanda_alg.QSVR import QSVR as qsvr_module


class _FakeAxes:
    def scatter(self, *args, **kwargs):
        return None

    def plot_surface(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def set_zlabel(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None


class _FakeFigure:
    def add_subplot(self, *args, **kwargs):
        return _FakeAxes()


def test_show_res_uses_each_features_own_plot_bounds(monkeypatch):
    predicted_points = []

    class FakeSVR:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, x, y):
            return self

        def predict(self, x):
            predicted_points.append(np.array(x, copy=True))
            return np.zeros(len(x))

    monkeypatch.setattr(qsvr_module, "SVR", FakeSVR)
    monkeypatch.setattr(qsvr_module.plt, "figure", lambda *args, **kwargs: _FakeFigure())
    monkeypatch.setattr(qsvr_module.plt, "show", lambda: None)

    qsvr = qsvr_module.Quantum_SVR.__new__(qsvr_module.Quantum_SVR)
    qsvr.x = np.array(
        [
            [-3.0, 10.0],
            [1.5, 20.0],
            [4.0, 14.0],
        ]
    )
    qsvr.y = np.array([0.0, 1.0, 2.0])

    qsvr.show_res()

    assert len(predicted_points) == 1
    grid = predicted_points[0]
    assert grid.shape == (900, 2)
    assert grid[:, 0].min() == -3.0
    assert grid[:, 0].max() == 4.0
    assert grid[:, 1].min() == 10.0
    assert grid[:, 1].max() == 20.0
