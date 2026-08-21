# Harness setup

This project starts with a human brief and an untouched repository clone.

If `harness/prepared.json` does not exist, run `/skill:build-harness`. During that setup
turn, the target repository is read-only: inspect and execute it, but do not edit,
format, commit, or switch branches. Create the project-local harness under `harness/`
and adjust `.omp/` instructions when useful. Do not access or modify the
auto-gpu-kernel source checkout.

Kopt runs the generated quick and full paths on the pristine clone and creates
`harness/prepared.json`. Once that file exists, normal optimization may edit the target
repository and may improve the project-local harness under the rules below.
