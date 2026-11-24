"""Per-block commitment table and Merkle root over an encoded array.

leaf[j] = BLAKE2b-256(key = s0, person = "memory_accounting/leaf", B'[j])   over the raw bytes of encoded block j
node    = BLAKE2b-256(person = "memory_accounting/node", left || right)      (an odd node is promoted unchanged)
root    = BLAKE2b-256(person = "memory_accounting/root", u64le(number of leaves) || top node)
Leaves and inner nodes hash in separate domains, so an inner node never passes for a leaf, and the root
binds the leaf count, so no tree of another size (say, its own inner nodes taken as leaves, or a tree
whose odd last node was promoted) has the same root. Before the tags, the leaves (a, b, c) and the
"leaves" (H(a || b), c) gave one root. In this stage the verifier recomputes the arrays and checks the
table; later the table alone (or a polynomial commitment) is what the verifier keeps.
"""

from __future__ import annotations

import hashlib

import numpy as np

LEAF_BYTES = 32
LEAF_TAG, NODE_TAG, ROOT_TAG = b"memory_accounting/leaf", b"memory_accounting/node", b"memory_accounting/root"


def _leaf(data, key: bytes) -> bytes:
    return hashlib.blake2b(data, key=key, digest_size=LEAF_BYTES, person=LEAF_TAG).digest()


def block_leaves(theta: np.ndarray, block_bytes: int, key: bytes) -> np.ndarray:
    """(nb, 32) uint8 array of keyed block digests. theta: uint64 array, length a multiple of the block."""
    data = np.ascontiguousarray(theta).view(np.uint8)
    nb = data.nbytes // block_bytes
    if nb * block_bytes != data.nbytes:
        raise ValueError("array length is not a multiple of the block")
    mv = memoryview(data)
    leaves = np.empty((nb, LEAF_BYTES), dtype=np.uint8)
    for j in range(nb):
        leaves[j] = np.frombuffer(_leaf(mv[j * block_bytes : (j + 1) * block_bytes], key), dtype=np.uint8)
    return leaves


def leaves_of(words: np.ndarray, block_bytes: int, key: bytes) -> np.ndarray:
    """`block_leaves` for an array that need not be whole blocks: the last block's digest is over what
    there is of it (Θ' and its image in a stream-decode bundle end where the weights end)."""
    data = np.ascontiguousarray(words).view(np.uint8).ravel()
    nb = -(-data.nbytes // block_bytes)
    mv = memoryview(data)
    leaves = np.empty((nb, LEAF_BYTES), dtype=np.uint8)
    for j in range(nb):
        leaves[j] = np.frombuffer(_leaf(mv[j * block_bytes : (j + 1) * block_bytes], key), dtype=np.uint8)
    return leaves


def merkle_root(leaves: np.ndarray) -> bytes:
    level = [bytes(leaf) for leaf in leaves]
    n = len(level)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(hashlib.blake2b(level[i] + level[i + 1], digest_size=LEAF_BYTES, person=NODE_TAG).digest())
            else:
                nxt.append(level[i])
        level = nxt
    return hashlib.blake2b(
        n.to_bytes(8, "little") + (level[0] if level else b""), digest_size=LEAF_BYTES, person=ROOT_TAG
    ).digest()
