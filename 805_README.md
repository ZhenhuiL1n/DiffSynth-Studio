# 805 Demo Quickstart

This folder contains a lightweight demo for the Wan2.2 TI2V pipeline. The full installation instructions, dependency list, and training details are documented in the **original README** (`README.md` in this repo). Please follow that file to set up Python, torch, LPIPS, opencv, etc., before running the demo below.

---

## 1. Download Checkpoints

All demo checkpoints are packaged here:

```
https://drive.google.com/file/d/1Pe4czIOyF1HObUzdxFHtWe5Qqt8wSrFi/view?usp=sharing
```

1. Download and unzip that archive.
2. Place the resulting `.safetensors` files inside the local `checkpoints/` directory (this repo already has `checkpoints/{body_best_ckpt, ewc_best, replay_best}` placeholders).

Credit: these checkpoints reuse the Wan2.2 TI2V backbone and LoRA training code from the official DiffSynth repo and the original authors’ training pipeline.

---

## 2. Test Images

Sample prompts/images live under `test_examples/`:

- `test_examples/ewc_body/*.png` – body OOD examples rendered at **832×480**. Use these when showcasing body EWC checkpoints (`checkpoints/ewc_best/*.safetensors`) or the body baseline.
- `test_examples/others/*.png` – mixed head/body prompts (768×768). Use these for head demos, replay experiments, etc.

---

## 3. Running the Demo

We provide a minimal CLI (`demo.py`) that renders 4 sample inputs for any checkpoint.

### Head demo (768×768)

```bash
python demo.py \
    --mode head \
    --ckpt checkpoints/ewc_best/ewc.safetensors \
    --device cuda \
    --output_dir demo_output
```

- Uses `test_examples/others/ood_head_*.png`.
- Writes MP4s under `demo_output/head/`.

### Body demo (832×480 EWC body prompts)

```bash
python demo.py \
    --mode body \
    --ckpt checkpoints/body_best_ckpt/8.safetensors \
    --device cuda \
    --output_dir demo_output
```

- Uses the `test_examples/ewc_body/*.png` images (832×480).
- Make sure your checkpoint corresponds to the same resolution (e.g., EWC body or body best).

Swap `--ckpt` with any of the downloaded checkpoints to showcase different models (e.g., `checkpoints/replay_best/replay.safetensors`).

---

## 4. Notes

- All dependencies (torch, torchvision, lpips, opencv, etc.) follow the versions listed in the original README.
- The demo only covers inference; evaluation scripts and training utilities remain in the repo but aren’t required for quick experiments.
- Please credit the original DiffSynth/Wan2.2 TI2V authors when sharing results.

Enjoy exploring the 805 demo!
