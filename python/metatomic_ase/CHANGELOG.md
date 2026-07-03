# Changelog

All notable changes to metatomic-ase are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/metatensor/metatomic/)

### Added

- `MetatomicCalculator` now resolves and exposes an `energy_ensemble` output
  (respecting `variants`), and gained a new `compute_energy_ensemble_gradients()`
  method computing conservative forces/stress ensembles as the Jacobian of the
  `energy_ensemble` output with respect to positions/strain.

<!-- Possible sections
### Fixed

### Changed

### Removed
-->

## [Version 0.1.1](https://github.com/metatensor/metatomic/releases/tag/metatomic-ase-v0.1.1) - 2026-05-13

### Added

- `non_conservative` in `MetatomicCalculator` now also accepts `"forces"` and
  `"stress"`. `"forces"` reads forces directly from the model while still
  computing stress via autograd; `"stress"` does the reverse.

## [Version 0.1.0](https://github.com/metatensor/metatomic/releases/tag/metatomic-ase-v0.1.0) - 2026-03-25

- `metatomic-ase` is now a standalone package, containing the ASE integration
  for metatomic models.
