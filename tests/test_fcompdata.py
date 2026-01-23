"""Basic tests for fcompdata package."""

import numpy as np
import pytest


class TestImports:
    """Test that package imports correctly."""

    def test_import_package(self):
        import fcompdata
        assert hasattr(fcompdata, "__version__")

    def test_import_datasets(self):
        from fcompdata import M1, M3, Tourism
        assert M1 is not None
        assert M3 is not None
        assert Tourism is not None

    def test_import_loaders(self):
        from fcompdata import load_m1, load_m3, load_tourism
        assert callable(load_m1)
        assert callable(load_m3)
        assert callable(load_tourism)

    def test_import_classes(self):
        from fcompdata import MCompDataset, MCompSeries
        assert MCompDataset is not None
        assert MCompSeries is not None


class TestM3Dataset:
    """Test M3 dataset loading and access."""

    def test_load_m3(self):
        from fcompdata import load_m3
        m3 = load_m3()
        assert len(m3) == 3003

    def test_lazy_m3_length(self):
        from fcompdata import M3
        assert len(M3) == 3003

    def test_m3_keys(self):
        from fcompdata import M3
        keys = M3.keys()
        assert keys[0] == 1  # 1-based indexing
        assert len(keys) == 3003

    def test_m3_first_series(self):
        from fcompdata import M3
        series = M3[1]
        assert series is not None
        assert series.sn is not None

    def test_m3_subset(self):
        from fcompdata import M3
        yearly = M3.subset("yearly")
        assert len(yearly) == 645


class TestM1Dataset:
    """Test M1 dataset loading and access."""

    def test_load_m1(self):
        from fcompdata import load_m1
        m1 = load_m1()
        assert len(m1) == 1001

    def test_lazy_m1_length(self):
        from fcompdata import M1
        assert len(M1) == 1001

    def test_m1_first_series(self):
        from fcompdata import M1
        series = M1[1]
        assert series is not None


class TestTourismDataset:
    """Test Tourism dataset loading and access."""

    def test_load_tourism(self):
        from fcompdata import load_tourism
        tourism = load_tourism()
        assert len(tourism) == 1311

    def test_lazy_tourism_length(self):
        from fcompdata import Tourism
        assert len(Tourism) == 1311

    def test_tourism_first_series(self):
        from fcompdata import Tourism
        series = Tourism[1]
        assert series is not None


class TestMCompSeries:
    """Test MCompSeries attributes and access patterns."""

    @pytest.fixture
    def sample_series(self):
        from fcompdata import M3
        return M3[1]

    def test_series_has_required_attributes(self, sample_series):
        assert hasattr(sample_series, "sn")
        assert hasattr(sample_series, "x")
        assert hasattr(sample_series, "xx")
        assert hasattr(sample_series, "h")
        assert hasattr(sample_series, "period")
        assert hasattr(sample_series, "type")
        assert hasattr(sample_series, "n")
        assert hasattr(sample_series, "description")

    def test_series_x_is_numpy_array(self, sample_series):
        assert isinstance(sample_series.x, np.ndarray)

    def test_series_xx_is_numpy_array(self, sample_series):
        assert isinstance(sample_series.xx, np.ndarray)

    def test_series_h_is_int(self, sample_series):
        assert isinstance(sample_series.h, int)
        assert sample_series.h > 0

    def test_series_n_matches_x_length(self, sample_series):
        assert sample_series.n == len(sample_series.x)

    def test_series_dict_style_access(self, sample_series):
        # Both attribute and dict-style access should work
        assert sample_series["x"] is sample_series.x
        assert sample_series["xx"] is sample_series.xx
        assert sample_series["h"] == sample_series.h
        assert sample_series["sn"] == sample_series.sn

    def test_series_keys(self, sample_series):
        keys = sample_series.keys()
        assert "x" in keys
        assert "xx" in keys
        assert "h" in keys
        assert "sn" in keys
        assert "period" in keys
        assert "type" in keys
        assert "n" in keys
        assert "description" in keys

    def test_series_repr(self, sample_series):
        repr_str = repr(sample_series)
        assert "MCompSeries" in repr_str
        assert sample_series.sn in repr_str


class TestMCompDataset:
    """Test MCompDataset operations."""

    @pytest.fixture
    def dataset(self):
        from fcompdata import load_m3
        return load_m3()

    def test_dataset_getitem_valid(self, dataset):
        series = dataset[1]
        assert series is not None

    def test_dataset_getitem_invalid(self, dataset):
        with pytest.raises(KeyError):
            _ = dataset[99999]

    def test_dataset_iteration(self, dataset):
        count = 0
        for series in dataset:
            count += 1
            if count >= 5:  # Only check first 5 to keep test fast
                break
        assert count == 5

    def test_dataset_items(self, dataset):
        for idx, series in dataset.items():
            assert isinstance(idx, int)
            assert series is not None
            break  # Only check first one

    def test_dataset_repr(self, dataset):
        repr_str = repr(dataset)
        assert "M3" in repr_str
        assert "3003" in repr_str

    def test_subset_returns_dataset(self, dataset):
        from fcompdata import MCompDataset
        subset = dataset.subset("monthly")
        assert isinstance(subset, MCompDataset)

    def test_subset_filters_correctly(self, dataset):
        monthly = dataset.subset("monthly")
        for series in monthly:
            assert series.type == "monthly"
            break  # Only check first one


class TestM4Imports:
    """Test M4 imports and error handling."""

    def test_import_m4(self):
        from fcompdata import M4
        assert M4 is not None

    def test_import_load_m4(self):
        from fcompdata import load_m4
        assert callable(load_m4)

    def test_load_m4_without_download_raises_error(self):
        from fcompdata import load_m4
        from fcompdata.download import get_m4_path
        # Skip if data already exists (e.g., from previous test runs)
        if get_m4_path("hourly") is not None:
            pytest.skip("M4 hourly data already downloaded")
        # M4 requires downloading first, so this should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_m4("hourly")

    def test_load_m4_invalid_frequency(self):
        from fcompdata import load_m4
        with pytest.raises(ValueError, match="Unknown frequency"):
            load_m4("invalid")


class TestDownloadModule:
    """Test download module functions."""

    def test_import_download_module(self):
        from fcompdata.download import download_m4, get_data_home, get_m4_path, clear_cache
        assert callable(download_m4)
        assert callable(get_data_home)
        assert callable(get_m4_path)
        assert callable(clear_cache)

    def test_get_data_home_returns_path(self):
        from pathlib import Path
        from fcompdata.download import get_data_home
        data_home = get_data_home()
        assert isinstance(data_home, Path)
        assert data_home.name == ".fcompdata"

    def test_get_m4_path_returns_none_when_not_downloaded(self):
        from fcompdata.download import get_m4_path
        # Should return None if not downloaded
        path = get_m4_path("yearly")
        # Path is None or exists (if previously downloaded)
        assert path is None or path.exists()

    def test_get_m4_path_invalid_frequency(self):
        from fcompdata.download import get_m4_path
        with pytest.raises(ValueError, match="Unknown frequency"):
            get_m4_path("invalid")

    def test_m4_urls_defined(self):
        from fcompdata.download import M4_URLS, M4_FILENAMES
        expected_frequencies = ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"]
        for freq in expected_frequencies:
            assert freq in M4_URLS
            assert freq in M4_FILENAMES

    def test_download_m4_invalid_frequency(self):
        from fcompdata.download import download_m4
        with pytest.raises(ValueError, match="Unknown frequency"):
            download_m4("invalid")
