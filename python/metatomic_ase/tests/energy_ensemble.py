from typing import Dict, List, Optional

import ase.build
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap, detach_block

import metatomic_lj_test
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    register_autograd_neighbors,
)
from metatomic_ase import MetatomicCalculator


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


@pytest.fixture
def model():
    return metatomic_lj_test.lennard_jones_model(
        atomic_type=28,
        cutoff=CUTOFF,
        sigma=SIGMA,
        epsilon=EPSILON,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )


@pytest.fixture
def atoms():
    np.random.seed(0xDEADBEEF)

    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.6, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.2 * np.random.rand(*atoms.positions.shape)

    return atoms


@pytest.mark.parametrize("non_conservative", [True, False])
def test_mean_matches_conservative(model, atoms, non_conservative):
    """The ensemble mean should match the calculator's own conservative forces/stress,
    regardless of whether the calculator itself runs in non_conservative mode: energy
    ensemble gradients have no non-conservative equivalent, they always use autograd."""
    calc = MetatomicCalculator(
        model,
        device="cpu",
        check_consistency=True,
        non_conservative=non_conservative,
        uncertainty_threshold=None,
    )

    result = calc.compute_energy_ensemble_gradients(atoms)
    assert set(result.keys()) == {"forces", "stress"}
    assert result["forces"].shape == (len(atoms), 3, 16)
    assert result["stress"].shape == (3, 3, 16)

    atoms = atoms.copy()
    atoms.calc = MetatomicCalculator(
        model, device="cpu", check_consistency=True, uncertainty_threshold=None
    )
    conservative_forces = atoms.get_forces()
    conservative_stress = atoms.get_stress(voigt=False)

    assert np.allclose(result["forces"].mean(axis=2), conservative_forces, atol=1e-4)
    assert np.allclose(result["stress"].mean(axis=2), conservative_stress, atol=1e-4)


def test_matches_manual_autograd(model, atoms):
    """Cross-check each ensemble member's forces/stress against an independent,
    manually-differentiated call to `run_model`, to catch any indexing/transpose bugs
    in the Jacobian extraction."""
    calc = MetatomicCalculator(
        model, device="cpu", check_consistency=True, uncertainty_threshold=None
    )
    result = calc.compute_energy_ensemble_gradients(atoms)

    ensemble_key = calc._energy_ensemble_key
    n_ensemble = result["forces"].shape[2]

    eval_options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={
            ensemble_key: ModelOutput(unit="eV", sample_kind="system"),
            calc._energy_key: ModelOutput(unit="eV", sample_kind="system"),
        },
        selected_atoms=None,
    )

    for member in [0, n_ensemble - 1]:
        positions = torch.tensor(atoms.positions, dtype=calc._dtype, requires_grad=True)
        strain = torch.eye(3, dtype=calc._dtype, requires_grad=True)
        cell = torch.tensor(np.array(atoms.cell), dtype=calc._dtype) @ strain
        types = torch.tensor(atoms.numbers, dtype=torch.int32)
        pbc = torch.tensor(atoms.pbc)

        # build the neighbor list without autograd, then re-attach it to the
        # differentiable positions/cell above, same trick as `calculate()`
        base_system = System(types, positions.detach(), cell.detach(), pbc)
        [base_system] = calc._nl_calculators.compute(systems=[base_system])

        system = System(types, positions @ strain, cell, pbc)
        for options in base_system.known_neighbor_lists():
            neighbors = detach_block(base_system.get_neighbor_list(options))
            register_autograd_neighbors(system, neighbors, True)
            system.add_neighbor_list(options, neighbors)

        outputs = calc._model([system], eval_options, check_consistency=True)
        energy_member = outputs[ensemble_key].block().values[0, member]
        energy_member.backward()

        manual_forces = -positions.grad.detach().numpy()
        manual_stress = strain.grad.detach().numpy() / atoms.cell.volume

        assert np.allclose(manual_forces, result["forces"][:, :, member], atol=1e-5)
        assert np.allclose(manual_stress, result["stress"][:, :, member], atol=1e-5)


def test_partial_computation(model, atoms):
    calc = MetatomicCalculator(
        model, device="cpu", check_consistency=True, uncertainty_threshold=None
    )

    forces_only = calc.compute_energy_ensemble_gradients(
        atoms, compute_forces=True, compute_stress=False
    )
    assert set(forces_only.keys()) == {"forces"}

    stress_only = calc.compute_energy_ensemble_gradients(
        atoms, compute_forces=False, compute_stress=True
    )
    assert set(stress_only.keys()) == {"stress"}


def test_errors(model, atoms):
    calc = MetatomicCalculator(
        model, device="cpu", check_consistency=True, uncertainty_threshold=None
    )

    with pytest.raises(ValueError, match="at least one of"):
        calc.compute_energy_ensemble_gradients(
            atoms, compute_forces=False, compute_stress=False
        )

    with pytest.raises(TypeError, match="only supports a single"):
        calc.compute_energy_ensemble_gradients([atoms])


def test_non_periodic_stress_warns(model, atoms):
    calc = MetatomicCalculator(
        model, device="cpu", check_consistency=True, uncertainty_threshold=None
    )
    non_periodic = atoms.copy()
    non_periodic.pbc = False

    with pytest.warns(UserWarning, match="likely to be wrong"):
        calc.compute_energy_ensemble_gradients(non_periodic, compute_stress=True)


def test_variant(model, atoms):
    calc = MetatomicCalculator(
        model, device="cpu", check_consistency=True, uncertainty_threshold=None
    )
    result = calc.compute_energy_ensemble_gradients(atoms)

    calc_doubled = MetatomicCalculator(
        model,
        device="cpu",
        check_consistency=True,
        variants={"energy": "doubled"},
        uncertainty_threshold=None,
    )
    assert calc_doubled._energy_ensemble_key == "energy_ensemble/doubled"
    result_doubled = calc_doubled.compute_energy_ensemble_gradients(atoms)

    assert np.allclose(2.0 * result["forces"], result_doubled["forces"], atol=1e-4)
    assert np.allclose(2.0 * result["stress"], result_doubled["stress"], atol=1e-4)


class _EnergyOnlyModel(torch.nn.Module):
    """Minimal model exposing only a plain 'energy' output, no 'energy_ensemble'."""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        device = systems[0].device
        dtype = systems[0].positions.dtype
        samples = Labels(
            ["system"], torch.arange(len(systems), device=device).reshape(-1, 1)
        )
        block = TensorBlock(
            values=torch.zeros((len(systems), 1), dtype=dtype, device=device),
            samples=samples,
            components=[],
            properties=Labels("energy", torch.tensor([[0]], device=device)),
        )
        return {"energy": TensorMap(Labels("_", torch.tensor([[0]])), [block])}


@pytest.fixture
def model_without_ensemble():
    return AtomisticModel(
        _EnergyOnlyModel().eval(),
        ModelMetadata(),
        ModelCapabilities(
            outputs={
                "energy": ModelOutput(sample_kind="system", unit="eV"),
            },
            atomic_types=[28],
            interaction_range=0.0,
            length_unit="Angstrom",
            supported_devices=["cpu"],
            dtype="float64",
        ),
    )


def test_missing_energy_ensemble_output(model_without_ensemble, atoms):
    calc = MetatomicCalculator(
        model_without_ensemble,
        device="cpu",
        check_consistency=True,
        uncertainty_threshold=None,
    )
    assert calc._energy_ensemble_key is None

    with pytest.raises(ValueError, match="does not have an 'energy_ensemble' output"):
        calc.compute_energy_ensemble_gradients(atoms)
