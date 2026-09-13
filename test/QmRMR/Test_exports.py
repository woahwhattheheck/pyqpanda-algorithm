def test_star_import_exports_feature_selection():
    namespace = {}

    exec("from pyqpanda_alg.QmRMR import *", namespace)

    assert "Feature_Selection" in namespace
    assert namespace["Feature_Selection"].__name__ == "Feature_Selection"
