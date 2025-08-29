# R11: EDG Epsilon Constraints

This report infers epsilon from R8 perihelion bootstrap means using epsilon_hat = A0_boot_mean / GR.

## Per Integrator
| planet | integrator | epsilon_hat | epsilon_std | A0_boot_mean | A0_boot_std | gr_rad_per_orbit |
| --- | --- | --- | --- | --- | --- | --- |
| Earth | rk4 | 0.999992 | 3.881567e-06 | 1.861103e-07 | 7.224056e-13 | 1.861119e-07 |
| Earth | vv | 0.797157 | 0.0729511 | 1.483603e-07 | 1.357707e-08 | 1.861119e-07 |
| Mercury | rk4 | 1 | 1.000000e-12 | 5.020401e-07 | 0 | 5.020398e-07 |
| Mercury | vv | 1.00241 | 1.873596e-04 | 5.032509e-07 | 9.406195e-11 | 5.020398e-07 |
| Venus | rk4 | 1 | 6.068867e-07 | 2.572646e-07 | 1.561299e-13 | 2.572637e-07 |
| Venus | vv | 0.64635 | 0.12627 | 1.662823e-07 | 3.248472e-08 | 2.572637e-07 |

## Per Planet Consensus
| planet | epsilon_consensus | epsilon_consensus_std | n_integrators |
| --- | --- | --- | --- |
| Earth | 0.999992 | 3.881567e-06 | 2 |
| Mercury | 1 | 1.000000e-12 | 2 |
| Venus | 1 | 6.068867e-07 | 2 |

## Global Consensus

- epsilon_global = 1 +/- 1.000000e-12 (68%)
- 68% CI: [1, 1]
- 95% CI: [1, 1]

