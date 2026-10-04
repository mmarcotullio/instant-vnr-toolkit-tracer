# Recorded per-dataset hyperparameters

`train.py` takes `--trace-tmax`, `--trace-vc-max-correction-frac`, and `--grad-clip-norm` as
plain values. This file records the validated values for every dataset used in this
project, so each one can be reproduced exactly without re-deriving anything.

For a new dataset, run `tools/calibrate_hyperparams.py` to get a starting point for
`--trace-tmax` and `--trace-vc-max-correction-frac`. `--grad-clip-norm` doesn't need a fresh
derivation unless you observe instability.

**The datasets themselves (tornado, lifted, ammonia, isotropic8192) are not included in or
linked from this repository**. The reproduction commands below use placeholder paths
and are included for transparency about exactly what was run and with what hyperparameters,
not as runnable commands. To use this method on your own data, substitute your own
vector field's path/dims, and use `tools/calibrate_hyperparams.py` to pick
`--trace-tmax` and `--trace-vc-max-correction-frac`.

## Values

| Dataset | dims | `--trace-tmax` | `--trace-vc-max-correction-frac` | `--grad-clip-norm` |
|---|---|---|---|---|
| tornado | 128 128 128 | 0.1944 | 0.0493 | 2.0 |
| lifted | 506 400 100 | 0.1091 | 0.0477 | 2.0 |
| ammonia | 864 240 640 | 0.0442 | 0.0458 | 2.0 |
| isotropic8192 | 640 640 640 | 0.1507 | 0.0517 | 2.0 |

## Reproduction commands

### tornado

```bash
# No-trace baseline
python train.py --filename /path/to/tornado.raw --dims 128 128 128 --n-channels 3 \
    --epochs 64 --expname tornado_notrace

# Trace-loss
python train.py --filename /path/to/tornado.raw --dims 128 128 128 --n-channels 3 \
    --train-traces --trace-tmax 0.1944 --epochs 64 --expname tornado_traceloss

# VC fine-tune (from the no-trace checkpoint)
python train.py --filename /path/to/tornado.raw --dims 128 128 128 --n-channels 3 \
    --train-traces --trace-tmax 0.1944 --trace-velocity-correction \
    --trace-vc-max-correction-frac 0.0493 --trace-warmup-epochs 0 \
    --init-checkpoint outputs/tornado_notrace_best.pt --epochs 3 --lr 1e-4 \
    --expname tornado_vc_finetune
```

### lifted (combustion)

```bash
# No-trace baseline
python train.py --filename /path/to/lifted.raw --dims 506 400 100 --n-channels 3 \
    --epochs 180 --expname lifted_notrace

# Trace-loss
python train.py --filename /path/to/lifted.raw --dims 506 400 100 --n-channels 3 \
    --train-traces --trace-tmax 0.1091 --epochs 180 --expname lifted_traceloss

# VC fine-tune
python train.py --filename /path/to/lifted.raw --dims 506 400 100 --n-channels 3 \
    --train-traces --trace-tmax 0.1091 --trace-velocity-correction \
    --trace-vc-max-correction-frac 0.0477 --trace-warmup-epochs 0 \
    --init-checkpoint outputs/lifted_notrace_best.pt --epochs 2 --lr 1e-4 \
    --expname lifted_vc_finetune
```

### ammonia

```bash
# No-trace baseline
python train.py --filename /path/to/ammonia.raw \
    --dims 864 240 640 --n-channels 3 --epochs 64 --expname ammonia_notrace

# Trace-loss
python train.py --filename /path/to/ammonia.raw \
    --dims 864 240 640 --n-channels 3 --train-traces --trace-tmax 0.0442 \
    --epochs 64 --expname ammonia_traceloss

# VC fine-tune
python train.py --filename /path/to/ammonia.raw \
    --dims 864 240 640 --n-channels 3 --train-traces --trace-tmax 0.0442 \
    --trace-velocity-correction --trace-vc-max-correction-frac 0.0458 \
    --trace-warmup-epochs 0 --init-checkpoint outputs/ammonia_notrace_best.pt \
    --epochs 3 --lr 1e-4 --expname ammonia_vc_finetune
```

### isotropic8192

```bash
# No-trace baseline
python train.py --filename /path/to/isotropic8192.raw \
    --dims 640 640 640 --n-channels 3 --epochs 64 --expname isotropic8192_notrace

# Trace-loss
python train.py --filename /path/to/isotropic8192.raw \
    --dims 640 640 640 --n-channels 3 --train-traces --trace-tmax 0.1507 \
    --epochs 64 --expname isotropic8192_traceloss

# VC fine-tune
python train.py --filename /path/to/isotropic8192.raw \
    --dims 640 640 640 --n-channels 3 --train-traces --trace-tmax 0.1507 \
    --trace-velocity-correction --trace-vc-max-correction-frac 0.0517 \
    --trace-warmup-epochs 0 --init-checkpoint outputs/isotropic8192_notrace_best.pt \
    --epochs 3 --lr 1e-4 --expname isotropic8192_vc_finetune
```

Add `--seed N` to reproduce a specific seed (controls the numpy/torch RNG, which jitters the
GT streamline seed-pool positions and trace-batch sampling); all other flags are identical
across seeds for a given dataset.
