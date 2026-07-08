"""Tests for read_probe_dat in ember.ts3 (ported from turbigen helmr branch).

Covers loading a Turbostream 3 probe ``.dat`` file into an ember ``Block``,
HDF5 caching, Fortran-order reshaping, and age-based deletion of the raw
``.dat``. Gas properties and sampling frequency are read from the
``probe_meta.yaml`` sidecar (no ``input.hdf5`` is required).
"""

import time
from unittest.mock import patch

import numpy as np

import pytest

from ember import util_yaml
from ember.ts3 import read_probe_dat, read_probe_dat_dir


def sampling_av(nsteps, **overrides):
    """Sampling vars whose implied time-sample count equals ``nsteps``.

    ``nt = (ncycle*nstep_cycle - nstep_save_start_probe) // nstep_save_probe``.
    Pass overrides to deliberately produce an inconsistent count.
    """
    av = {
        "ncycle": 1,
        "nstep_cycle": nsteps,
        "nstep_save_probe": 1,
        "nstep_save_start_probe": 0,
    }
    av.update(overrides)
    return av


def write_probe_meta(
    tmp_path,
    bid,
    pid,
    shape,
    *,
    cp=1005.0,
    ga=1.4,
    mu=1.8e-5,
    fs=7200.0,
    nsteps=10,
    av_overrides=None,
):
    """Write a probe_meta.yaml entry carrying geometry, gas props, sampling."""
    entry = {
        "shape": list(shape),
        "Omega": 0.0,
        "Nb": 10,
        "label": "test_probe",
        "cp": cp,
        "ga": ga,
        "mu": mu,
        "fs": fs,
        **sampling_av(nsteps, **(av_overrides or {})),
    }
    util_yaml.write_yaml({bid: {pid: entry}}, str(tmp_path / "probe_meta.yaml"))


def create_probe_dat(filepath, shape, nsteps=10):
    """Create a synthetic probe .dat file with recognisable data.

    Returns a dict of expected values for verification.
    """
    nspatial = int(np.prod(shape))

    # x: 0..nspatial-1, r: 0.5, rt: r*theta
    x = np.arange(nspatial, dtype=np.float32)
    r = np.full(nspatial, 0.5, dtype=np.float32)
    rt = r * np.linspace(0, 2 * np.pi, nspatial, dtype=np.float32)

    ro = np.full(nspatial, 1.2, dtype=np.float32)
    rovx = ro * 10.0  # Vx = 10 m/s
    rovr = ro * 0.0  # Vr = 0
    rorvt = ro * r * 5.0  # Vt = 5 m/s
    u = 208333.33
    Vx, Vr, Vt = rovx / ro, rovr / ro, rorvt / (ro * r)
    ke = 0.5 * (Vx**2 + Vr**2 + Vt**2)
    roe = ro * (u + ke)

    conserved = np.stack([x, r, rt, ro, rovx, rovr, rorvt, roe])

    # Repeat over time with a small density/energy variation
    steps = []
    for istep in range(nsteps):
        c = conserved.copy()
        factor = 1.0 + 0.01 * np.sin(2 * np.pi * istep / nsteps)
        c[3:8] *= factor
        steps.append(c.T)  # (nspatial, 8)
    data_all = np.concatenate(steps, axis=0)

    with open(filepath, "w") as f:
        f.write("x r rt ro rovx rovr rorvt roe\n")
        np.savetxt(f, data_all, fmt="%.8e")

    return {"Vx": float(Vx[0]), "nsteps": nsteps, "nspatial": nspatial}


def test_basic_dat_loading(tmp_path):
    shape = [5, 2]
    nsteps = 10
    bid, pid = 75, 16
    fs = 7200.0

    write_probe_meta(tmp_path, bid, pid, shape, fs=fs)
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    expected = create_probe_dat(dat_file, shape, nsteps)

    F, fs_out = read_probe_dat(str(dat_file))

    assert F is not None
    assert fs_out == fs
    assert F.shape == tuple(shape + [nsteps])

    # HDF5 cache was created
    assert (tmp_path / f"output_probe_{bid}_{pid}.hdf5").exists()

    # Recognisable data
    assert F.x[0, 0, 0] == 0.0
    assert F.x[1, 0, 0] == 1.0
    assert np.abs(F.rho[0, 0, 0] - 1.2) < 0.1
    assert np.abs(F.Vx[0, 0, 0] - expected["Vx"]) < 0.1


def test_cache_reuse(tmp_path):
    """Cache is created and reused; raw .dat deleted once old."""
    shape = [3, 3]
    nsteps = 10
    bid, pid = 76, 17

    write_probe_meta(tmp_path, bid, pid, shape)
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    create_probe_dat(dat_file, shape, nsteps)

    # Mock the .dat as 49 h old so it is deleted after first load
    old_time = time.time() - (49 * 3600)
    with patch("os.path.getmtime", return_value=old_time):
        F1, fs1 = read_probe_dat(str(dat_file))

    assert (tmp_path / f"output_probe_{bid}_{pid}.hdf5").exists()
    assert not dat_file.exists(), ".dat should be deleted when >48 h old"

    # Subsequent reads load from the cache even though the .dat is gone
    F2, fs2 = read_probe_dat(str(dat_file))
    F3, fs3 = read_probe_dat(str(dat_file))

    assert F1.shape == F2.shape == F3.shape
    assert fs1 == fs2 == fs3
    np.testing.assert_allclose(F1.x, F2.x, rtol=1e-6)
    np.testing.assert_allclose(F1.rho, F2.rho, rtol=1e-6)
    np.testing.assert_allclose(F1.Vx, F3.Vx, rtol=1e-6)


def test_fortran_order_reshape(tmp_path):
    shape = [2, 3, 5]
    nsteps = 10
    bid, pid = 77, 18

    write_probe_meta(tmp_path, bid, pid, shape)
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    create_probe_dat(dat_file, shape, nsteps)

    F, _ = read_probe_dat(str(dat_file))

    assert F.shape == tuple(shape + [nsteps])

    # x was 0..29; Fortran-order reshape to (2,3,5): x[i,j,k] = i + 2*j + 6*k
    assert F.x[0, 0, 0, 0] == 0
    assert F.x[1, 0, 0, 0] == 1
    assert F.x[0, 1, 0, 0] == 2
    assert F.x[0, 0, 1, 0] == 6
    assert F.x[1, 1, 1, 0] == 9

    assert np.all(F.rho > 0)
    assert np.abs(F.Vx[0, 0, 0, 0] - 10.0) < 0.1

    # Internal energy positive everywhere
    u = F.rhoe / F.rho - 0.5 * (F.Vx**2 + F.Vr**2 + F.Vt**2)
    assert np.all(u > 0)


def test_skip_age_check(tmp_path):
    """skip_age_check deletes a fresh .dat."""
    shape = [2, 2]
    bid, pid = 80, 21

    write_probe_meta(tmp_path, bid, pid, shape)
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    create_probe_dat(dat_file, shape, 10)

    F1, fs1 = read_probe_dat(str(dat_file), skip_age_check=True)

    assert not dat_file.exists(), ".dat deleted when skip_age_check=True"
    assert (tmp_path / f"output_probe_{bid}_{pid}.hdf5").exists()

    # Still loadable from cache
    F2, fs2 = read_probe_dat(str(dat_file), skip_age_check=True)
    assert F1.shape == F2.shape
    assert fs1 == fs2
    np.testing.assert_allclose(F1.x, F2.x, rtol=1e-6)


def test_skip_age_check_default_false(tmp_path):
    """Default behaviour preserves a recent .dat."""
    shape = [2, 2]
    bid, pid = 81, 22

    write_probe_meta(tmp_path, bid, pid, shape)
    dat_file = tmp_path / f"output_probe_{bid}_{pid}.dat"
    create_probe_dat(dat_file, shape, 10)

    read_probe_dat(str(dat_file))

    assert dat_file.exists(), "fresh .dat preserved when skip_age_check=False"
    assert (tmp_path / f"output_probe_{bid}_{pid}.hdf5").exists()


# --- read_probe_dat_dir (item 4) -----------------------------------------


def write_multi_probe(tmp_path, probes, *, fs=7200.0, nsteps=6):
    """Write probe_meta.yaml and matching .dat files for several probes.

    ``probes`` is an iterable of ``(bid, pid, shape, label)`` tuples.
    """
    meta = {}
    for bid, pid, shape, label in probes:
        meta.setdefault(bid, {})[pid] = {
            "shape": list(shape),
            "Omega": 0.0,
            "Nb": 10,
            "label": label,
            "cp": 1005.0,
            "ga": 1.4,
            "mu": 1.8e-5,
            "fs": fs,
            **sampling_av(nsteps),
        }
        create_probe_dat(tmp_path / f"output_probe_{bid}_{pid}.dat", shape, nsteps)
    util_yaml.write_yaml(meta, str(tmp_path / "probe_meta.yaml"))


def test_dat_dir_loads_all(tmp_path):
    nsteps = 6
    probes = [
        (1, 0, [2], "inlet"),
        (1, 1, [3, 4], "outlet"),
        (2, 0, [2, 2, 2], "mid"),
    ]
    write_multi_probe(tmp_path, probes, nsteps=nsteps)

    blocks, fs = read_probe_dat_dir(tmp_path)

    assert fs == 7200.0
    assert len(blocks) == 3
    shapes = sorted(b.shape for b in blocks)
    assert shapes == sorted(tuple(s) + (nsteps,) for _, _, s, _ in probes)


def test_dat_dir_label_substring(tmp_path):
    write_multi_probe(
        tmp_path,
        [(1, 0, [2], "stator_te"), (1, 1, [2], "rotor_te"), (2, 0, [2], "stator_le")],
    )

    blocks, _ = read_probe_dat_dir(tmp_path, label="stator")
    assert len(blocks) == 2


def test_dat_dir_label_exact(tmp_path):
    write_multi_probe(
        tmp_path,
        [(1, 0, [2], "stator"), (1, 1, [2], "stator_te")],
    )

    blocks, _ = read_probe_dat_dir(tmp_path, label="stator", exact=True)
    assert len(blocks) == 1


def test_dat_dir_label_no_match_raises(tmp_path):
    write_multi_probe(tmp_path, [(1, 0, [2], "inlet"), (1, 1, [2], "outlet")])

    with pytest.raises(ValueError, match="inlet"):
        read_probe_dat_dir(tmp_path, label="nonexistent")


def test_dat_dir_empty_dir(tmp_path):
    assert read_probe_dat_dir(tmp_path) == ([], None)


# --- time-dimension validation (item 5) ----------------------------------


def test_time_dim_validates_correct(tmp_path):
    nsteps = 8
    write_probe_meta(tmp_path, 1, 0, [3], nsteps=nsteps)
    dat = tmp_path / "output_probe_1_0.dat"
    create_probe_dat(dat, [3], nsteps)

    F, _ = read_probe_dat(str(dat))
    assert F.shape == (3, nsteps)


def test_time_dim_mismatch_raises(tmp_path):
    nsteps = 8
    # av's imply nt = nsteps + 5, but the .dat has nsteps
    write_probe_meta(
        tmp_path, 1, 0, [3], nsteps=nsteps, av_overrides={"nstep_cycle": nsteps + 5}
    )
    dat = tmp_path / "output_probe_1_0.dat"
    create_probe_dat(dat, [3], nsteps)

    with pytest.raises(ValueError):
        read_probe_dat(str(dat))


def test_time_dim_validate_false_bypasses(tmp_path):
    nsteps = 8
    write_probe_meta(
        tmp_path, 1, 0, [3], nsteps=nsteps, av_overrides={"nstep_cycle": nsteps + 5}
    )
    dat = tmp_path / "output_probe_1_0.dat"
    create_probe_dat(dat, [3], nsteps)

    F, _ = read_probe_dat(str(dat), validate=False)
    assert F.shape == (3, nsteps)  # dimension inferred, no check


def test_time_dim_skipped_when_av_missing(tmp_path):
    nsteps = 8
    # nstep_save_probe=0 means the expected count is unknown -> no validation,
    # so a "wrong" nstep_cycle does not raise.
    write_probe_meta(
        tmp_path,
        1,
        0,
        [3],
        nsteps=nsteps,
        av_overrides={"nstep_save_probe": 0, "nstep_cycle": nsteps + 5},
    )
    dat = tmp_path / "output_probe_1_0.dat"
    create_probe_dat(dat, [3], nsteps)

    F, _ = read_probe_dat(str(dat))
    assert F.shape == (3, nsteps)


def test_dat_dir_validate_false_threads_through(tmp_path):
    # Probe whose sidecar count disagrees with the data; the dir loader must
    # honour validate=False and still return it.
    nsteps = 6
    bid, pid = 1, 0
    meta = {
        bid: {
            pid: {
                "shape": [2],
                "Omega": 0.0,
                "Nb": 10,
                "label": "p",
                "cp": 1005.0,
                "ga": 1.4,
                "mu": 1.8e-5,
                "fs": 7200.0,
                **sampling_av(nsteps, nstep_cycle=nsteps + 3),
            }
        }
    }
    util_yaml.write_yaml(meta, str(tmp_path / "probe_meta.yaml"))
    create_probe_dat(tmp_path / f"output_probe_{bid}_{pid}.dat", [2], nsteps)

    blocks, _ = read_probe_dat_dir(tmp_path, validate=False)
    assert len(blocks) == 1
