"""Test module for ember.average area-weighted averaging and integration functions.

Tests area-weighted averaging and integration functionality for flow analysis.

Test cases:
- test_block_2d: 2D cut from 3D block with flow through the cut plane
- test_node_to_face: Node to face distribution for averaging
- test_integrate_scalar_uniform_field: Scalar integration of uniform fields
- test_integrate_scalar_linear_field: Scalar integration of linear fields
- test_integrate_vector_zero_flux: Vector integration with zero net flux
- test_area_average: Area-weighted averaging calculations
- test_mass_average_uniform_field: Mass-weighted averaging of uniform fields
- test_mass_average_linear_field: Mass-weighted averaging of linear fields
- test_mass_average_vs_area_average_uniform_flow: Comparison of mass vs area averaging for uniform flow
- test_mass_average_varying_flow: Mass-weighted averaging with varying flow
- test_mass_average_zero_net_mass_flux: Mass averaging with zero net mass flux
- test_partial_integration: Partial integration over subdomains
- test_invalid_axes: Error handling for invalid integration axes
- test_integrate_scalar_triangulated: Scalar integration on triangulated meshes
- test_integrate_vector_triangulated: Vector integration on triangulated meshes
- test_total_area: Total area calculations
- test_total_area_triangulated: Total area calculations for triangulated meshes
- test_mix_out_structured: Mixing plane calculations on structured grids
- test_mix_out_triangulated_vs_structured: Comparison of triangulated vs structured mixing calculations
- test_mix_super: Supersonic mixing plane calculations
- test_mix_radial: Radial mixing calculations
- test_mix_out_k_axis_flip_invariance: Verify mixing is invariant to k-axis flip
- test_mix_out_k_axis_flip_invariance_negative_vt: Verify k-axis flip invariance with negative Vt
"""

import numpy as np
import pytest
from ember.block import Block
from ember import average, util
from ember.geometry import node_to_face_2d
import ember.fluid
import ember.average


@pytest.fixture
def test_block_2d():
    """Create a 2D cut from 3D block with flow through the cut plane."""
    # Create 3D block first
    shape = (5, 4, 6)
    xrt = util.linmesh3((0.0, 0.1), (0.9, 1.1), (0.0, 0.1), shape)
    block_3d = Block(shape=shape)
    block_3d.set_x(xrt[..., 0])
    block_3d.set_r(xrt[..., 1])
    block_3d.set_t(xrt[..., 2])

    # Set up fluid and flow state
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block_3d.set_fluid(fluid)
    block_3d.set_P_T(1e5, 300.0)
    block_3d.set_Vx(10.0)
    block_3d.set_Vr(0.0)
    block_3d.set_Vt(0.0)

    # Take a 2D cut at constant x (normal to flow direction)
    # This creates a cut with flow through the plane
    block_2d = block_3d[2]  # Cut at i=2 (constant x)

    return block_2d


def create_test_block():
    """Create a simple 2D test block with known geometry."""
    ni, nj = 5, 4
    L = 0.1

    # Create grid avoiding r=0
    xv = np.linspace(0, L, ni)
    yv = np.linspace(0.01, L, nj)
    x, y = np.meshgrid(xv, yv, indexing="ij")

    # Convert to polar
    r = np.sqrt(y**2)  # Just y since z=0
    t = np.zeros_like(x)  # Constant theta

    block = Block(shape=(ni, nj))
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    return block


def test_node_to_face():
    """Test that node_to_face averaging works correctly."""
    # Simple 3x3 nodal data
    nodal = np.arange(9, dtype=np.float32).reshape(3, 3)
    face_avg = node_to_face_2d(nodal)

    # Should be (2, 2) shape
    assert face_avg.shape == (2, 2)

    # Verify manual calculation for first face
    expected_00 = 0.25 * (nodal[0, 0] + nodal[1, 0] + nodal[1, 1] + nodal[0, 1])
    assert np.isclose(face_avg[0, 0], expected_00)


def test_integrate_scalar_uniform_field():
    """Test scalar integration with uniform field."""
    block = create_test_block()

    # Uniform scalar field
    uniform_value = 5.0
    nodal_scalar = np.full(block.shape, uniform_value, dtype=np.float32)
    face_scalar = node_to_face_2d(nodal_scalar)

    # Get face areas
    dA_face = block.dA_quad

    # Integrate over entire domain
    integral = average._integrate_scalar(face_scalar, dA_face, axes=(0, 1))

    # Should equal uniform_value * total_area
    expected = uniform_value * np.sum(util.vecnorm(dA_face))
    assert np.isclose(integral, expected, rtol=1e-6)


def test_integrate_scalar_linear_field():
    """Test scalar integration with linear field."""
    block = create_test_block()

    # Linear field: φ = x
    x_coords = block.x
    x_face = node_to_face_2d(x_coords)

    dA_face = block.dA_quad
    integral = average._integrate_scalar(x_face, dA_face, axes=(0, 1))

    # For linear field over rectangular domain, integral should be
    # field_at_centroid * total_area
    x_centroid = np.mean(x_coords)
    expected = x_centroid * np.sum(util.vecnorm(dA_face))
    assert np.isclose(integral, expected, rtol=1e-5)


def test_integrate_vector_zero_flux():
    """Test vector integration with zero net flux."""
    block = create_test_block()

    # Create a circulation field: V = [-y, x, 0] (no flux through boundary)
    x, y = block.x, block.r  # r is our y-coordinate
    zero_z = np.zeros_like(x)
    velocity = np.stack([-y, x, zero_z], axis=-1)
    velocity = node_to_face_2d(velocity)

    dA_face = block.dA_quad
    flux = average._integrate_vector(velocity, dA_face, axes=(0, 1))

    # Should be very close to zero for circulation field
    assert np.abs(flux) < 1e-10


def test_area_average(test_block_2d):
    """Test area-weighted averaging."""
    block = test_block_2d

    # Linear field
    nodal_scalar = block.x + 2.0 * block.r

    avg = average.area_average(nodal_scalar, block)

    # Should be close to the centroid value
    centroid_value = np.mean(nodal_scalar)
    assert np.isclose(avg, centroid_value, rtol=1e-2)


def test_mass_average_uniform_field(test_block_2d):
    """Test mass-weighted averaging with uniform field."""
    block = test_block_2d

    # Uniform scalar field
    uniform_value = 5.0
    nodal_scalar = np.full(block.shape, uniform_value, dtype=np.float32)

    avg = average.mass_average(nodal_scalar, block)

    # For uniform field, mass average should equal the uniform value
    assert np.isclose(avg, uniform_value, rtol=1e-6)


def test_mass_average_linear_field(test_block_2d):
    """Test mass-weighted averaging with linear field."""
    block = test_block_2d

    # Linear field in x-direction
    nodal_scalar = block.x

    avg = average.mass_average(nodal_scalar, block)

    # For uniform flow and linear field, result should be close to centroid value
    centroid_value = np.mean(nodal_scalar)
    assert np.isclose(avg, centroid_value, rtol=1e-3)


def test_mass_average_vs_area_average_uniform_flow(test_block_2d):
    """Test that mass and area averages are similar for uniform flow."""
    block = test_block_2d

    # Linear field
    nodal_scalar = block.x + 0.5 * block.r

    mass_avg = average.mass_average(nodal_scalar, block)
    area_avg = average.area_average(nodal_scalar, block)

    # For uniform flow, mass and area averages should be very similar
    assert np.isclose(mass_avg, area_avg, rtol=1e-2)


def test_mass_average_varying_flow(test_block_2d):
    """Test mass averaging with spatially varying flow."""
    block = test_block_2d

    # Create varying velocity field - higher velocity at higher r (radial variation)
    Vx_varying = 10.0 + 50.0 * (block.r - np.min(block.r)) / np.ptp(block.r)
    block.set_Vx(Vx_varying)
    block.set_Vr(0.0)
    block.set_Vt(0.0)

    # Scalar field that varies with r
    nodal_scalar = 1.0 + 2.0 * (block.r - np.min(block.r)) / np.ptp(block.r)

    mass_avg = average.mass_average(nodal_scalar, block)
    area_avg = average.area_average(nodal_scalar, block)

    # Mass average should be biased toward higher velocity regions (higher r)
    # So mass average should be larger than area average
    assert mass_avg > area_avg

    # Both should be reasonable values
    assert 1.0 < mass_avg < 3.0
    assert 1.0 < area_avg < 3.0


def test_mass_average_zero_net_mass_flux():
    """Test that mass_average raises error when net mass flux is zero."""
    # Create a closed 2D block (not a cut) where net mass flux should be zero
    ni, nj = 5, 4
    L = 0.1

    # Create grid avoiding r=0
    xv = np.linspace(0, L, ni)
    yv = np.linspace(0.01, L, nj)
    x, y = np.meshgrid(xv, yv, indexing="ij")

    # Convert to polar
    r = np.sqrt(y**2)  # Just y since z=0
    t = np.zeros_like(x)  # Constant theta

    block = Block(shape=(ni, nj))
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Set up fluid and flow state
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)
    block.set_P_T(1e5, 300.0)
    block.set_Vx(10.0)
    block.set_Vr(0.0)
    block.set_Vt(0.0)

    # Create a uniform scalar field
    nodal_scalar = np.ones(block.shape, dtype=np.float32)

    # Should raise ValueError due to zero net mass flux
    with pytest.raises(ValueError, match="Net mass flux through the block is zero"):
        average.mass_average(nodal_scalar, block)


def test_partial_integration():
    """Test integration over partial axes."""
    block = create_test_block()

    # Uniform field
    nodal_scalar = np.ones(block.shape, dtype=np.float32)
    dA_face = block.dA_quad
    face_scalar = node_to_face_2d(nodal_scalar)

    # Integrate over j-axis only (axes=(1,))
    integral_i = average._integrate_scalar(face_scalar, dA_face, axes=(1,))

    # Should have shape (ni-1,)
    assert integral_i.shape == (block.ni - 1,)
    assert np.all(integral_i > 0)


def test_invalid_axes():
    """Test that invalid axes raise errors."""
    # axes > 1 should error
    with pytest.raises(ValueError):
        average._get_axes((0, 2), triangulated=False)

    with pytest.raises(ValueError):
        average._get_axes((-1,), triangulated=False)

    with pytest.raises(ValueError):
        average._get_axes((), triangulated=False)

    with pytest.raises(ValueError):
        average._get_axes((), triangulated=True)

    with pytest.raises(ValueError):
        average._get_axes((1,), triangulated=True)

    with pytest.raises(ValueError):
        average._get_axes((0,), triangulated=True)

    with pytest.raises(ValueError):
        average._get_axes((0, 1), triangulated=True)


def test_integrate_scalar_triangulated():
    """Test integrate_scalar with triangulated data."""
    # Create simple triangulated data: 2 triangles
    nodal_scalar = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
    )  # (2, 3)

    # Create mock area vectors
    dA_face = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)  # (2, 3)
    face_scalar = nodal_scalar.mean(axis=1)  # Mean per triangle

    # Test integration
    result = average._integrate_scalar(face_scalar, dA_face, axes=(0,))

    # Expected: mean of triangle 1 vertices (2.0) * area mag (1.0) + mean of triangle 2 vertices (5.0) * area mag (0.5)
    expected = 2.0 * 1.0 + 5.0 * 0.5  # 2.0 + 2.5 = 4.5
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_integrate_vector_triangulated():
    """Test integrate_vector with triangulated data."""
    # Create triangulated vector data: 2 triangles, 3 vertices each, 3 vector components
    # Triangle 1: vertices have vectors [1,0,0], [0,1,0], [0,0,1]
    # Triangle 2: vertices have vectors [2,0,0], [0,2,0], [0,0,2]
    nodal_vector = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        ],
        dtype=np.float32,
    )  # Shape (2, 3, 3)
    face_vector = nodal_vector.mean(axis=1)  # Mean per triangle

    # Area vectors: triangle 1 in x direction, triangle 2 in y direction
    dA_face = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)  # (2, 3)

    # Test integration
    result = average._integrate_vector(face_vector, dA_face, axes=(0,))

    # Expected:
    # Triangle 1: mean vector [1/3, 1/3, 1/3] · [1,0,0] = 1/3
    # Triangle 2: mean vector [2/3, 2/3, 2/3] · [0,1,0] = 2/3
    # Total = 1/3 + 2/3 = 1.0
    expected = 1.0
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_total_area():
    """Test total_area function returns shape (3,) for structured cuts."""
    block = create_test_block()

    # Test structured grid
    result = average.total_area(block)
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    assert np.all(np.isfinite(result)), "All components should be finite"


def test_total_area_triangulated():
    """Test total_area function returns shape (3,) for triangulated cuts."""
    from ember.cut import triangulate_to_unstructured
    import ember.fluid

    # Create a 2D block
    block_2d = create_test_block()

    # Set up fluid and state for triangulation
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block_2d.set_fluid(fluid)
    block_2d.set_P_T(1e5, 300.0)
    block_2d.set_Vx(10.0)
    block_2d.set_Vr(0.0)
    block_2d.set_Vt(0.0)

    # Create triangulated block
    tri_block = triangulate_to_unstructured(block_2d)

    # Test triangulated grid
    result = average.total_area(tri_block)
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    assert np.all(np.isfinite(result)), "All components should be finite"


def test_mix_out_structured():
    """Test mix_out function with a uniform structured 2D cut."""
    # Use existing create_test_block function for proper 2D geometry

    shape = (7, 8, 9)
    xrt = util.linmesh3((0.0, 0.1), (0.9, 1.1), (0.01, 0.0), shape)
    block = Block(shape=shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)
    block.set_P_T(2e5, 400.0)
    block.set_conserved(
        np.array(
            [1.0785884e00, 7.3343224e01, 3.2785469e-01, -9.4219635e01, 1.7516456e03]
        )[None, None, :]
    )

    block = block[0]  # Take a 2D cut

    from ember.cut import triangulate_to_unstructured

    block = triangulate_to_unstructured(block)

    C = block

    print(C.conserved.mean(axis=(0, 1)))
    print(C.xrt.mean(axis=(0, 1)))
    print(ember.average.total_area(C))
    print(ember.average.flow_conserved(C))

    # Apply mix-out
    print("Mixing out structured block...")
    mixed = average.mix_out(block)

    # Verify mixed-out block is scalar (empty shape)
    assert mixed.ndim == 0, f"Mixed block should be scalar, got {mixed.ndim}D"
    assert mixed.shape == (), f"Mixed block should have empty shape, got {mixed.shape}"

    # Verify mixed state has reasonable values
    assert mixed.rho > 0, "Mixed density should be positive"
    assert mixed.P > 0, "Mixed pressure should be positive"
    assert mixed.T > 0, "Mixed temperature should be positive"
    assert np.isfinite(mixed.V), "Mixed velocity should be finite"


def test_mix_out_triangulated_vs_structured():
    """Test that mix_out gives similar results for structured vs triangulated cuts."""
    from ember.cut import triangulate_to_unstructured

    shape = (7, 8, 9)
    xrt = util.linmesh3((0.0, 0.1), (0.9, 1.1), (0.0, 0.1), shape)
    block = Block(shape=shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)
    block.set_P_T(2e5, 400.0)
    block.set_Vx(100.0)
    block.set_Vr(60.0)
    block.set_Vt(30.0)
    block.set_label("beans")

    block_structured = block[0]

    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block_structured.set_fluid(fluid)
    block_structured.set_P_T(1e5, 300.0)
    block_structured.set_Vx(100.0)
    block_structured.set_Vr(20.0)
    block_structured.set_Vt(10.0)

    # Mix out structured version
    print("Mixing out structured block...")
    mixed_structured = average.mix_out(block_structured)

    # Create triangulated version of same block
    block_triangulated = triangulate_to_unstructured(block_structured)

    # Mix out triangulated version
    print("Mixing out tri block...")
    mixed_triangulated = average.mix_out(block_triangulated)

    # Both should be scalar blocks
    assert mixed_structured.ndim == 0, "Structured mixed block should be scalar"
    assert mixed_triangulated.ndim == 0, "Triangulated mixed block should be scalar"

    # Results should be very similar (within 1% tolerance)
    rtol = 1e-2
    np.testing.assert_allclose(mixed_structured.rho, mixed_triangulated.rho, rtol=rtol)
    np.testing.assert_allclose(mixed_structured.P, mixed_triangulated.P, rtol=rtol)
    np.testing.assert_allclose(mixed_structured.T, mixed_triangulated.T, rtol=rtol)
    np.testing.assert_allclose(mixed_structured.Vx, mixed_triangulated.Vx, rtol=rtol)
    np.testing.assert_allclose(mixed_structured.Vr, mixed_triangulated.Vr, rtol=rtol)
    np.testing.assert_allclose(mixed_structured.Vt, mixed_triangulated.Vt, rtol=rtol)


def test_mix_super():
    shape = (7, 8, 9)
    xrt = util.linmesh3((0.0, 0.1), (0.9, 1.1), (0.0, 0.1), shape)
    block = Block(shape=shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)
    block.set_P_T(2e5, 400.0)
    block.set_Vx(500.0)
    block.set_Vr(0.0)
    block.set_Vt(0.0)
    assert (block.Ma > 1.0).all()

    cut = block[0]
    mix = average.mix_out(cut)
    assert mix.Ma > 1.0


def _low_mach_axial_cut(Vt=0.0):
    """Build a uniform low-Mach (Ma~0.1) cut for area-ratio tests."""
    shape = (7, 8, 9)
    xrt = util.linmesh3((0.0, 0.1), (0.9, 1.1), (0.0, 0.1), shape)
    block = Block(shape=shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)
    block.set_P_T(2e5, 400.0)
    block.set_Vx(40.0)
    block.set_Vr(0.0)
    block.set_Vt(Vt)
    return block[0]


def test_mix_out_area_ratio_unit_identity():
    """AR=1 must reproduce the default constant-area mixed-out state exactly."""
    cut = _low_mach_axial_cut()
    mix_default = average.mix_out(cut)
    mix_unit = average.mix_out(cut, AR=1.0)
    np.testing.assert_allclose(mix_unit.conserved, mix_default.conserved)


def test_mix_out_area_ratio_conservation():
    """The isentropic contraction conserves mass flux, ho, and entropy.

    With area scaled by AR, the meridional mass flux rho*Vm must scale as 1/AR
    (so rho*Vm*AR is invariant), while stagnation enthalpy and entropy are
    unchanged because the contraction is reversible.
    """
    cut = _low_mach_axial_cut()
    assert (cut.Ma < 0.12).all(), "Flow should be low Mach for this test"

    mix_ref = average.mix_out(cut, AR=1.0)
    rhoVm_ref = mix_ref.rho * mix_ref.Vm

    for AR in (1.1, 0.9, 0.5, 0.25):
        mix = average.mix_out(cut, AR=AR)
        # Mass conservation: rho*Vm scales as 1/AR
        np.testing.assert_allclose(mix.rho * mix.Vm * AR, rhoVm_ref, rtol=1e-4)
        # Reversible step: stagnation enthalpy and entropy are conserved
        np.testing.assert_allclose(mix.ho, mix_ref.ho, rtol=1e-4)
        np.testing.assert_allclose(mix.s, mix_ref.s, rtol=1e-4)


def test_mix_out_area_ratio_low_mach_velocity():
    """At low Mach (near-incompressible), velocity scales as 1/AR.

    Density is nearly constant for a small contraction, so mass conservation
    rho*Vm*(AR*A)=const forces Vm ~ Vm0/AR. The match is exact only in the
    incompressible limit, so a near-unity AR is used.
    """
    cut = _low_mach_axial_cut()
    mix_ref = average.mix_out(cut, AR=1.0)

    for AR in (0.95, 1.05):
        mix = average.mix_out(cut, AR=AR)
        np.testing.assert_allclose(mix.Vm, mix_ref.Vm / AR, rtol=5e-3)


def test_mix_out_area_ratio_conserves_angular_momentum():
    """Swirl velocity Vt is held constant through the contraction.

    At fixed radius, conserving angular momentum r*Vt means Vt is unchanged;
    only the meridional velocity adjusts to satisfy mass conservation.
    """
    cut = _low_mach_axial_cut(Vt=30.0)
    mix_ref = average.mix_out(cut, AR=1.0)
    assert mix_ref.Vt > 0.0

    for AR in (0.9, 0.5):
        mix = average.mix_out(cut, AR=AR)
        # Vt unchanged (angular momentum at fixed radius)
        np.testing.assert_allclose(mix.Vt, mix_ref.Vt, rtol=1e-3)
        # Meridional mass flux still scales as 1/AR
        np.testing.assert_allclose(
            mix.rho * mix.Vm * AR, mix_ref.rho * mix_ref.Vm, rtol=1e-4
        )


def test_mix_out_area_ratio_choke_raises():
    """A contraction strong enough to choke the meridional flow raises."""
    cut = _low_mach_axial_cut()
    # Inlet Ma~0.1, so contracting by ~10x drives the meridional flow past M=1.
    with pytest.raises(RuntimeError, match="chok"):
        average.mix_out(cut, AR=0.05)


def test_mix_radial():
    shape = (7, 8, 9)
    xrt = util.linmesh3((0.0, 0.1), (0.9, 1.1), (0.0, 0.1), shape)
    block = Block(shape=shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)
    block.set_P_T(2e5, 400.0)
    block.set_Vx(300.0)
    block.set_Vr(0.0)
    block.set_Vt(0.0)

    cut = block[:, 0, :]
    assert np.ptp(cut.r) < 1e-6, "Cut should be radial with constant r"
    mix = average.mix_out(cut)
    atol = util.get_atol(cut.conserved, mix.r, rtol=1e-5)
    for i in range(5):
        np.testing.assert_allclose(
            mix.conserved[i], cut.conserved[..., i], atol=atol[i]
        )

    block.set_Vx(0.0)
    block.set_Vr(-100.0)
    block.set_Vt(0.0)

    cut = block[:, 0, :]
    assert np.ptp(cut.r) < 1e-6, "Cut should be radial with constant r"
    mix = average.mix_out(cut)
    atol = util.get_atol(cut.conserved, mix.r, rtol=1e-5)
    for i in range(5):
        np.testing.assert_allclose(
            mix.conserved[i], cut.conserved[..., i], atol=atol[i]
        )

    block.set_Vx(0.0)
    block.set_Vr(300.0)
    block.set_Vt(0.0)

    cut = block[:, 0, :]
    assert np.ptp(cut.r) < 1e-6, "Cut should be radial with constant r"
    mix = average.mix_out(cut)
    atol = util.get_atol(cut.conserved, mix.r, rtol=1e-5)
    for i in range(5):
        np.testing.assert_allclose(
            mix.conserved[i], cut.conserved[..., i], atol=atol[i]
        )


def test_mix_out_k_axis_flip_invariance():
    """Test that mixing out a structured block gives identical results when k-axis is flipped.

    The mixed-out state should be independent of the ordering of the k-axis
    (circumferential direction) since we're integrating over that dimension.
    """
    # Create a 3D block with non-uniform flow
    shape = (7, 8, 9)
    xrt = util.linmesh3((0.0, 0.1), (0.9, 1.1), (0.0, 2 * np.pi), shape)
    block = Block(shape=shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set up fluid and non-uniform flow state
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)

    # Create spatially varying flow to make the test more challenging
    # Vary pressure, temperature, and velocity with position
    P_base = 1e5
    T_base = 300.0
    P_field = P_base * (1.0 + 0.2 * np.sin(2 * np.pi * block.x / 0.1))
    T_field = T_base * (1.0 + 0.1 * np.cos(2 * np.pi * block.r))
    block.set_P_T(P_field, T_field)

    # Spatially varying velocity
    Vx = 100.0 + 20.0 * np.sin(2 * np.pi * block.t)
    Vr = 10.0 * np.cos(2 * np.pi * block.x / 0.1)
    Vt = 5.0 * np.sin(2 * np.pi * block.r)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Take a 2D cut at constant x
    block_cut = block[3]

    # Mix out the original cut
    mixed_original = average.mix_out(block_cut)

    # Flip the k-axis
    block_cut_flipped = block_cut[:, ::-1]

    # Verify the flip worked (coordinates should be reversed in k)
    assert np.allclose(block_cut.t[:, 0], block_cut_flipped.t[:, -1]), (
        "k-axis flip should reverse theta coordinates"
    )

    # Mix out the flipped cut
    mixed_flipped = average.mix_out(block_cut_flipped)

    # Both should be scalar blocks
    assert mixed_original.ndim == 0, "Original mixed block should be scalar"
    assert mixed_flipped.ndim == 0, "Flipped mixed block should be scalar"

    # Results should be identical (or nearly so, within numerical precision)
    # Use tight tolerance since this is a fundamental symmetry
    # Note: mixing calculation involves iteration, so small numerical differences can accumulate
    rtol = 5e-6
    atol = 1e-8

    np.testing.assert_allclose(
        mixed_original.rho,
        mixed_flipped.rho,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed density should be identical for flipped k-axis",
    )
    np.testing.assert_allclose(
        mixed_original.P,
        mixed_flipped.P,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed pressure should be identical for flipped k-axis",
    )
    np.testing.assert_allclose(
        mixed_original.T,
        mixed_flipped.T,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed temperature should be identical for flipped k-axis",
    )
    np.testing.assert_allclose(
        mixed_original.Vx,
        mixed_flipped.Vx,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed Vx should be identical for flipped k-axis",
    )
    np.testing.assert_allclose(
        mixed_original.Vr,
        mixed_flipped.Vr,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed Vr should be identical for flipped k-axis",
    )
    np.testing.assert_allclose(
        mixed_original.Vt,
        mixed_flipped.Vt,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed Vt should be identical for flipped k-axis",
    )

    # Also verify conserved variables for completeness
    np.testing.assert_allclose(
        mixed_original.conserved,
        mixed_flipped.conserved,
        rtol=rtol,
        atol=atol,
        err_msg="All mixed conserved variables should be identical for flipped k-axis",
    )


def test_mix_out_k_axis_flip_invariance_negative_vt():
    """Test k-axis flip invariance with negative circumferential velocity.

    Verifies that mixing out gives identical results even when Vt is negative,
    which exercises the sign-handling in the mixing algorithm.
    """
    # Create a 3D block
    shape = (7, 8, 9)
    xrt = util.linmesh3((0.0, 0.1), (0.9, 1.1), (0.0, 2 * np.pi), shape)
    block = Block(shape=shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set up fluid and flow state with negative Vt
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)

    # Create spatially varying flow with predominantly negative Vt
    P_base = 1e5
    T_base = 300.0
    P_field = P_base * (1.0 + 0.15 * np.sin(2 * np.pi * block.x / 0.1))
    T_field = T_base * (1.0 + 0.08 * np.cos(2 * np.pi * block.r))
    block.set_P_T(P_field, T_field)

    # Negative circumferential velocity (opposite swirl direction)
    Vx = 120.0 + 15.0 * np.cos(2 * np.pi * block.t)
    Vr = 8.0 * np.sin(2 * np.pi * block.x / 0.1)
    Vt = -50.0 + 10.0 * np.sin(2 * np.pi * block.r)  # Negative Vt
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Verify Vt is indeed negative on average
    assert np.mean(Vt) < 0, "Test should have negative average Vt"

    # Take a 2D cut at constant x
    block_cut = block[4]

    # Mix out the original cut
    mixed_original = average.mix_out(block_cut)

    # Flip the k-axis
    block_cut_flipped = block_cut[:, ::-1]

    # Mix out the flipped cut
    mixed_flipped = average.mix_out(block_cut_flipped)

    # Both should be scalar blocks
    assert mixed_original.ndim == 0, "Original mixed block should be scalar"
    assert mixed_flipped.ndim == 0, "Flipped mixed block should be scalar"

    # Results should be identical
    rtol = 5e-6
    atol = 1e-8

    np.testing.assert_allclose(
        mixed_original.rho,
        mixed_flipped.rho,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed density should be identical with negative Vt",
    )
    np.testing.assert_allclose(
        mixed_original.P,
        mixed_flipped.P,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed pressure should be identical with negative Vt",
    )
    np.testing.assert_allclose(
        mixed_original.T,
        mixed_flipped.T,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed temperature should be identical with negative Vt",
    )
    np.testing.assert_allclose(
        mixed_original.Vx,
        mixed_flipped.Vx,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed Vx should be identical with negative Vt",
    )
    np.testing.assert_allclose(
        mixed_original.Vr,
        mixed_flipped.Vr,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed Vr should be identical with negative Vt",
    )
    np.testing.assert_allclose(
        mixed_original.Vt,
        mixed_flipped.Vt,
        rtol=rtol,
        atol=atol,
        err_msg="Mixed Vt should be identical with negative Vt",
    )

    # Verify the mixed Vt is negative (preserves sign)
    assert mixed_original.Vt < 0, "Mixed Vt should remain negative"
    assert mixed_flipped.Vt < 0, "Mixed Vt (flipped) should remain negative"

    # Also verify conserved variables
    np.testing.assert_allclose(
        mixed_original.conserved,
        mixed_flipped.conserved,
        rtol=rtol,
        atol=atol,
        err_msg="All mixed conserved variables should be identical with negative Vt",
    )


def test_mix_out_reference_invariance():
    """Test that mix_out gives same dimensional result with different references.

    The mixed-out state should be physically identical regardless of the
    nondimensionalisation (L_ref, rho_ref, V_ref) and datum (P_dtm, T_dtm)
    chosen, since it is a physical quantity. The second case uses a high T_dtm
    so that the mean stagnation enthalpy ho of the input data is negative,
    exercising the solver with negative internal energy values.
    """
    shape = (7, 8, 9)
    xrt = util.linmesh3((0.0, 0.1), (0.9, 1.1), (0.0, 0.1), shape)

    # --- Baseline: default references (L_ref=1, rho_ref=1, V_ref=1) ---
    block_base = Block(shape=shape)
    block_base.set_x(xrt[..., 0])
    block_base.set_r(xrt[..., 1])
    block_base.set_t(xrt[..., 2])
    fluid_base = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block_base.set_fluid(fluid_base)
    block_base.set_P_T(2e5, 400.0)
    block_base.set_Vx(100.0)
    block_base.set_Vr(20.0)
    block_base.set_Vt(10.0)
    cut_base = block_base[0]
    mixed_base = average.mix_out(cut_base)

    # --- Same physical state with different references and datum ---
    L_ref = 0.05
    rho_ref = 1.5
    V_ref = 200.0
    T_dtm = 800.0  # High enough that ho = cp*(T - T_dtm/gamma) + V^2/2 < 0
    P_dtm = 2e5

    block_ref = Block(shape=shape)
    block_ref.set_x(xrt[..., 0])
    block_ref.set_r(xrt[..., 1])
    block_ref.set_t(xrt[..., 2])
    fluid_ref = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=1e-5,
        Pr=0.72,
        rho_ref=rho_ref,
        V_ref=V_ref,
        P_dtm=P_dtm,
        T_dtm=T_dtm,
    )
    block_ref.set_fluid(fluid_ref)
    block_ref.set_P_T(2e5, 400.0)
    block_ref.set_Vx(100.0)
    block_ref.set_Vr(20.0)
    block_ref.set_Vt(10.0)
    block_ref.set_L_ref(L_ref)
    cut_ref = block_ref[0]

    # Verify that ho is negative with this datum
    assert np.all(cut_ref.ho < 0.0), (
        f"Expected negative ho with T_dtm={T_dtm}, got mean ho={cut_ref.ho.mean()}"
    )

    mixed_ref = average.mix_out(cut_ref)

    # Dimensional results should match
    rtol = 1e-4
    np.testing.assert_allclose(
        mixed_base.rho,
        mixed_ref.rho,
        rtol=rtol,
        err_msg="Density should be invariant to reference values",
    )
    np.testing.assert_allclose(
        mixed_base.P,
        mixed_ref.P,
        rtol=rtol,
        err_msg="Pressure should be invariant to reference values",
    )
    np.testing.assert_allclose(
        mixed_base.T,
        mixed_ref.T,
        rtol=rtol,
        err_msg="Temperature should be invariant to reference values",
    )
    np.testing.assert_allclose(
        mixed_base.Vx,
        mixed_ref.Vx,
        rtol=rtol,
        err_msg="Vx should be invariant to reference values",
    )
    np.testing.assert_allclose(
        mixed_base.Vr,
        mixed_ref.Vr,
        rtol=rtol,
        err_msg="Vr should be invariant to reference values",
    )
    np.testing.assert_allclose(
        mixed_base.Vt,
        mixed_ref.Vt,
        rtol=rtol,
        err_msg="Vt should be invariant to reference values",
    )


if __name__ == "__main__":
    test_mix_radial()
