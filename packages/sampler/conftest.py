import sys

# Prevent the local pysampler/ source directory from shadowing the installed
# package when pytest is run from inside packages/sampler/.
sys.path = [p for p in sys.path if p not in ("", ".")]
