"""Regression tests for optional CLI overrides that used to be read from `args` instead of `cfg`.

`--batch-size`, `--grad-accum`, `--lr` and `--epochs` are optional overrides: the preset
supplies the value and the argument stays None. Reading `args.batch_size` directly therefore
produced `None * 2` and a TypeError *during training*, after the model had already spent
minutes on the first epoch. These tests pin the resolved values to the preset.

NumPy/pandas only, no torch: they check the resolution logic, not a training run.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import finetune_transformer as ft  # noqa: E402
import pseudo_label as pl  # noqa: E402


def resolve(preset: str, **overrides) -> dict:
    """The same resolution pseudo_label.train_pseudo() performs on cfg."""
    cfg = dict(ft.PRESETS[preset])
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


class OptionalOverrideResolution(unittest.TestCase):
    def test_pseudo_label_defaults_to_general_pretrained_init(self):
        ns = pl.parse_args(["--data-dir", "data", "--stage", "train", "--pseudo", "p.csv",
                            "--base", "runs/deberta", "--preset", "large", "--output", "runs/x"])
        self.assertFalse(ns.init_from_base)
        src = Path(pl.__file__).read_text(encoding="utf-8")
        self.assertIn("init_source = str(base_dir) if args.init_from_base else cfg[\"model\"]", src)

    def test_unset_overrides_fall_back_to_the_preset(self):
        ns = pl.parse_args(["--data-dir", "data", "--stage", "train",
                            "--pseudo", "p.csv", "--preset", "large", "--output", "runs/x"])
        self.assertIsNone(ns.batch_size)
        self.assertIsNone(ns.grad_accum)
        self.assertIsNone(ns.lr)
        self.assertIsNone(ns.epochs)
        cfg = resolve(ns.preset, batch_size=ns.batch_size, grad_accum=ns.grad_accum,
                      lr=ns.lr, epochs=ns.epochs)
        self.assertEqual(cfg["batch_size"], ft.PRESETS["large"]["batch_size"])
        self.assertEqual(cfg["grad_accum"], ft.PRESETS["large"]["grad_accum"])
        self.assertIsNotNone(cfg["lr"])
        self.assertIsNotNone(cfg["epochs"])

    def test_eval_batch_size_is_never_none_for_any_preset(self):
        """The evaluation batch size must come from cfg, i.e. it exists even without --batch-size."""
        for preset in ft.PRESETS:
            cfg = resolve(preset)
            eval_bs = int(min(64, max(8, 2 * cfg["batch_size"])))
            self.assertIsInstance(eval_bs, int)
            self.assertGreater(eval_bs, 0)
            self.assertLessEqual(eval_bs, 64)

    def test_explicit_override_still_wins(self):
        ns = pl.parse_args(["--data-dir", "data", "--stage", "train", "--pseudo", "p.csv",
                            "--preset", "large", "--output", "runs/x", "--batch-size", "2"])
        cfg = resolve(ns.preset, batch_size=ns.batch_size)
        self.assertEqual(cfg["batch_size"], 2)

    def test_documented_command_line_resolves(self):
        """The exact 1-2 command from the guide must produce a complete config."""
        ns = pl.parse_args(["--data-dir", "data", "--stage", "train", "--pseudo", "pseudo/pseudo.csv",
                            "--base", "runs/deberta", "--preset", "large", "--output", "runs/deberta-pl",
                            "--weight", "0.5", "--epochs", "1", "--lr", "5e-6"])
        cfg = resolve(ns.preset, batch_size=ns.batch_size, grad_accum=ns.grad_accum,
                      lr=ns.lr, epochs=ns.epochs)
        self.assertEqual(cfg["epochs"], 1.0)
        self.assertEqual(cfg["lr"], 5e-6)
        self.assertEqual(cfg["batch_size"], ft.PRESETS["large"]["batch_size"])
        self.assertEqual(cfg["grad_accum"], ft.PRESETS["large"]["grad_accum"])

    def test_source_does_not_multiply_a_optional_argument(self):
        """Guards the actual bug: `args.batch_size * 2` in the evaluation path."""
        src = Path(pl.__file__).read_text(encoding="utf-8")
        for bad in ("args.batch_size * 2", "args.batch_size*2",
                    "args.grad_accum *", "args.lr *", "args.epochs *"):
            self.assertNotIn(bad, src, f"{bad!r} multiplies an optional override, which is None "
                                       f"when the preset supplies the value")


if __name__ == "__main__":
    unittest.main(verbosity=2)
