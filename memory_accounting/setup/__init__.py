"""PoSM setup phase (offline, untimed, run by both parties).

Prover:   weights -> codec-group payloads -> [+ verifier-seeded filler] -> per-GPU VDE encoding
          -> per-block commitment table + Merkle root -> manifest.
Verifier: weights + manifest -> recompute every encoded array -> check commitments -> keep the arrays
          (or the tables) for challenge-time recomputation; then baseline the honest RTT.

Scope (LVHEM_MACC prototype, stage 1): the verifier knows the weights, the encoding is enforced by
recomputation, two GPUs, hardware facts trusted. Confidential (ZK) commitments are future work.
"""
