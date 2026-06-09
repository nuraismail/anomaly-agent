import numpy as np
import pytest

from scripts.validate_simulations import bin_spectra, dl_from_cl, make_ell_bins


def test_dl_from_cl_applies_ell_factor_after_monopole_and_dipole():
    dl = dl_from_cl(np.asarray([10.0, 10.0, 1.0, 2.0]))

    assert dl[0] == 0.0
    assert dl[1] == 0.0
    assert dl[2] == pytest.approx(3.0 / np.pi)
    assert dl[3] == pytest.approx(12.0 / np.pi)


def test_make_ell_bins_covers_final_partial_bin():
    assert make_ell_bins(2, 8, 3) == [(2, 4), (5, 7), (8, 8)]


def test_bin_spectra_uses_two_ell_plus_one_weights():
    spectra = np.asarray([[0.0, 10.0, 20.0, 30.0, 40.0]])
    binned, centers = bin_spectra(spectra, [(1, 2), (3, 4)])

    assert binned.shape == (1, 2)
    assert binned[0, 0] == pytest.approx(np.average([10.0, 20.0], weights=[3.0, 5.0]))
    assert binned[0, 1] == pytest.approx(np.average([30.0, 40.0], weights=[7.0, 9.0]))
    np.testing.assert_allclose(
        centers,
        [
            np.average([1.0, 2.0], weights=[3.0, 5.0]),
            np.average([3.0, 4.0], weights=[7.0, 9.0]),
        ],
    )
