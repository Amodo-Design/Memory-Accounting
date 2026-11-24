"""Process-wide VDE variant for the full-throughput benchmark and its verifier."""
import os

FIELD_BITS = int(os.environ.get("POSM_VDE_BITS", "89"))
TWEAK = os.environ.get("POSM_VDE_TWEAK", "group64")
if FIELD_BITS not in (61, 89) or TWEAK not in ("element", "group64"):
    raise ValueError("Expected POSM_VDE_BITS=61|89 and POSM_VDE_TWEAK=element|group64")
PROFILE = f"p{FIELD_BITS}_{TWEAK}"
GROUP_RAW, GROUP_ENC = (15, 16) if FIELD_BITS == 61 else (55, 64)
FLAGS = [f"-DPOSM_VDE_BITS={FIELD_BITS}", f"-DPOSM_GROUP64={int(TWEAK == 'group64')}"]

LAYOUT = os.environ.get("POSM_VDE_LAYOUT", "legacy")
if LAYOUT not in ("legacy", "packed128") or (LAYOUT == "packed128" and (FIELD_BITS, TWEAK) != (89, "element")):
    raise ValueError("packed128 requires 89-bit element tweaks")
PACKED128 = LAYOUT == "packed128"
if PACKED128:
    PROFILE += "_packed128"
    GROUP_RAW, GROUP_ENC = 121, 128  # eight rows, word-aligned plaintext for weight streaming
    FLAGS.append("-DPOSM_PACKED128=1")
ROW_WORDS89 = 16 if PACKED128 else 8
