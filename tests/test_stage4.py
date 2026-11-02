"""Tests for the stage-4 helpers (LLM trainer helpers, run_cv --trainer llm, ensemble_cv combiners,
compare_runs). NumPy/pandas only: no torch, no GPU, no model download.

What these tests do NOT cover: anything inside finetune_llm.main() that needs torch / transformers /
peft (model loading, LoRA, training, saving, re-loading). That path is checked only by the
``--smoke`` run on the GPU machine.

    py -3.12 tests/test_stage4.py        (from the submission folder)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import assemble_cv as ac  # noqa: E402
import compare_runs as cr  # noqa: E402
import ensemble_cv as ec  # noqa: E402
import finetune_llm as fl  # noqa: E402
import run_cv as rc  # noqa: E402

LABELS = ("negative", "positive")


# --------------------------------------------------------------------------- finetune_llm helpers
class ModelNameGuard(unittest.TestCase):
    def test_accepts_base_checkpoints(self):
        for name in ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen2.5-7B",
                     "meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.3", "ibm-granite/granite-3.3-8b-base",
                     r"models\Qwen3-8B-Base", "models/Qwen3-8B-Base"):
            fl.check_model_name(name)  # must not raise

    def test_refuses_tuned_or_label_trained_checkpoints(self):
        for name in ("Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it",
                     "Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Reranker-8B", "intfloat/e5-mistral-7b-instruct",
                     "siebert/sentiment-roberta-large-english", "lvwerra/distilbert-imdb",
                     "meta-llama/Llama-3.1-8B-Instruct", "HuggingFaceH4/zephyr-7b-sft-full",
                     r"models\Qwen3-8B", "BAAI/bge-large-en"):
            with self.assertRaises(ValueError, msg=name):
                fl.check_model_name(name)

    def test_only_the_last_two_path_parts_are_inspected(self):
        # an unrelated folder higher up (here "chat") must not trigger the guard; backslashes count too
        fl.check_model_name(r"downloads\chat\models\Qwen3-8B-Base")

    def test_presets_are_all_acceptable(self):
        for p in fl.PRESETS.values():
            fl.check_model_name(p["model"], allow_post_trained=bool(p.get("post_trained", False)))

    def test_only_one_preset_is_post_trained(self):
        self.assertEqual([k for k, p in fl.PRESETS.items() if p.get("post_trained")], ["qwen3.8-27b"])

    def test_auto_preset(self):
        self.assertEqual(fl.auto_preset(23.9), "qwen3-8b")
        self.assertEqual(fl.auto_preset(16.0), "qwen3-4b")
        self.assertEqual(fl.auto_preset(8.0), "qwen3-1.7b")
        self.assertEqual(fl.auto_preset(95.6), "qwen3-8b")  # 14B is never chosen implicitly

    def test_qwen3_14b_preset_is_the_8b_recipe_on_the_14b_base(self):
        p14, p8 = fl.PRESETS["qwen3-14b"], fl.PRESETS["qwen3-8b"]
        self.assertEqual(p14["model"], "Qwen/Qwen3-14B-Base")
        self.assertFalse(p14["grad_ckpt"])  # the memory probe switches it on if the GPU is too small
        self.assertEqual(set(p14), set(p8))
        for key in set(p8) - {"model", "grad_ckpt"}:
            self.assertEqual(p14[key], p8[key], key)
        fl.check_model_name("Qwen/Qwen3-14B-Base")  # a local download is accepted too

    def test_mistral_24b_preset_is_the_8b_recipe_on_the_mistral_base(self):
        pm, p8 = fl.PRESETS["mistral-24b"], fl.PRESETS["qwen3-8b"]
        self.assertEqual(pm["model"], "mistralai/Mistral-Small-24B-Base-2501")
        self.assertTrue(pm["grad_ckpt"])  # 47 GB of weights + a 32,768-wide MLP: no room without it
        self.assertEqual(pm["tokenizer_kwargs"], {"fix_mistral_regex": True})
        self.assertEqual(set(pm) - {"tokenizer_kwargs"}, set(p8))
        for key in set(p8) - {"model", "grad_ckpt"}:
            self.assertEqual(pm[key], p8[key], key)
        fl.check_model_name("mistralai/Mistral-Small-24B-Base-2501")
        json.dumps(pm)  # the preset is saved in llm_meta.json / run_info.json

    def test_instruct_sibling_of_the_mistral_base_is_refused(self):
        with self.assertRaises(ValueError):
            fl.check_model_name("mistralai/Mistral-Small-24B-Instruct-2501")

    def test_qwen38_27b_preset_is_the_8b_recipe(self):
        pq, p8 = fl.PRESETS["qwen3.8-27b"], fl.PRESETS["qwen3-8b"]
        self.assertEqual(pq["model"], "Qwen/Qwen3.8-27B")
        self.assertTrue(pq["grad_ckpt"])
        self.assertEqual((pq["post_trained"], pq["loader"], pq["lora_targets"]), (True, "vlm", "text-linear"))
        self.assertEqual(set(pq) - {"post_trained", "loader", "lora_targets"}, set(p8))
        for key in set(p8) - {"model", "grad_ckpt"}:
            self.assertEqual(pq[key], p8[key], key)
        json.dumps(pq)

    def test_post_trained_qwen38_needs_the_explicit_flag(self):
        for name in ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"):
            with self.assertRaises(ValueError, msg=name):
                fl.check_model_name(name)                     # still refused by default
            fl.check_model_name(name, allow_post_trained=True)  # allowed for the stage-7 preset

    def test_the_flag_does_not_open_other_post_trained_checkpoints(self):
        for name in ("Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3.8-27B-Instruct", "Qwen/Qwen3.5-27B",
                     "JonathanColetti/Qwen3.8-27B-Uncensored", "cyankiwi/Qwen3.8-27B-AWQ-INT4",
                     "siebert/sentiment-roberta-large-english", "lvwerra/distilbert-imdb"):
            with self.assertRaises(ValueError, msg=name):
                fl.check_model_name(name, allow_post_trained=True)


class LoraTargets(unittest.TestCase):
    # module names in the layout of a Qwen3.5-family vision-language checkpoint (text backbone taken out)
    NAMES = ["layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z", "layers.0.linear_attn.in_proj_b",
             "layers.0.linear_attn.in_proj_a", "layers.0.linear_attn.out_proj", "layers.0.mlp.gate_proj",
             "layers.0.mlp.up_proj", "layers.0.mlp.down_proj", "layers.3.self_attn.q_proj",
             "layers.3.self_attn.k_proj", "layers.3.self_attn.v_proj", "layers.3.self_attn.o_proj",
             "layers.3.mlp.gate_proj", "visual.blocks.0.attn.qkv", "visual.merger.mlp.0", "mtp.layers.0.mlp.up_proj",
             "lm_head", "model.visual.blocks.1.mlp.linear_fc1"]

    def test_text_layers_kept_vision_mtp_and_head_dropped(self):
        kept = fl.select_lora_targets(self.NAMES)
        self.assertEqual(kept, self.NAMES[:13])
        summary = fl.target_summary(kept)
        self.assertEqual(summary["gate_proj"], 2)
        self.assertEqual(summary["in_proj_qkv"], 1)
        self.assertNotIn("qkv", summary)

    def test_whole_component_match_only(self):
        # "visualize" / "mtp_like" are not the vision tower or the MTP head
        self.assertEqual(fl.select_lora_targets(["layers.0.visualize_proj", "layers.0.mtp_like"]),
                         ["layers.0.visualize_proj", "layers.0.mtp_like"])

    def test_nothing_left_is_an_error(self):
        with self.assertRaises(ValueError):
            fl.select_lora_targets(["visual.blocks.0.attn.qkv", "lm_head"])

    def test_find_text_backbone(self):
        class Box:
            def __init__(self, **kw):
                self.__dict__.update(kw)
        lm = Box(layers=[1, 2])
        self.assertIs(fl.find_text_backbone(Box(model=Box(language_model=lm, visual=Box()))), lm)  # transformers 5
        self.assertIs(fl.find_text_backbone(Box(language_model=lm)), lm)                          # older layout
        with self.assertRaises(RuntimeError):
            fl.find_text_backbone(Box(model=Box(visual=Box())))

    def test_hidden_of_reads_text_config(self):
        class C:
            pass
        flat, nested, inner = C(), C(), C()
        flat.hidden_size = 4096
        inner.hidden_size = 5120
        nested.text_config = inner
        self.assertEqual(fl.hidden_of(flat), 4096)
        self.assertEqual(fl.hidden_of(nested), 5120)
        with self.assertRaises(ValueError):
            fl.hidden_of(C())


class BuildInput(unittest.TestCase):
    def test_short_review_is_kept_whole_with_prompt(self):
        seq = fl.build_input([5, 6, 7], [1, 2], [8, 9], None, max_len=64, head=16)
        self.assertEqual(seq.tolist(), [1, 2, 5, 6, 7, 8, 9])

    def test_bos_is_prepended(self):
        seq = fl.build_input([5, 6], [1], [9], 100, max_len=64, head=16)
        self.assertEqual(seq.tolist(), [100, 1, 5, 6, 9])

    def test_long_review_is_head_tail_truncated_and_prompt_survives(self):
        review = list(range(1000, 1300))           # 300 tokens
        prefix, suffix = [1, 2, 3], [7, 8, 9, 10]
        seq = fl.build_input(review, prefix, suffix, 0, max_len=128, head=40).tolist()
        self.assertEqual(len(seq), 128)
        self.assertEqual(seq[:4], [0, 1, 2, 3])
        self.assertEqual(seq[-4:], suffix)
        body = seq[4:-4]
        self.assertEqual(len(body), 128 - 8)
        self.assertEqual(body[:40], review[:40])     # head
        self.assertEqual(body[40:], review[-(len(body) - 40):])  # tail

    def test_too_small_max_len_is_an_error(self):
        with self.assertRaises(ValueError):
            fl.build_input([1, 2, 3], list(range(20)), list(range(20)), None, max_len=48, head=8)


class Batching(unittest.TestCase):
    def _lengths(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        return np.clip(rng.lognormal(5.5, 0.6, n).astype(int), 20, 1024)

    def test_plan_covers_every_row_once_and_respects_the_budget(self):
        L = self._lengths()
        idx = np.arange(0, len(L), 2)                # a subset, like a training fold
        plan = fl.plan_updates(L, idx, 16, 4096, 16, np.random.default_rng(1))
        seen = np.concatenate([m for upd in plan for m in upd])
        self.assertEqual(sorted(seen.tolist()), idx.tolist())
        self.assertEqual(len(plan), int(np.ceil(len(idx) / 16)))
        for upd in plan:
            self.assertLessEqual(sum(len(m) for m in upd), 16)
            for m in upd:
                self.assertLessEqual(len(m), 16)
                if len(m) > 1:
                    self.assertLessEqual(len(m) * L[m].max(), 4096)
                # the first member is the longest, which is what the packing assumes
                self.assertEqual(L[m[0]], L[m].max())

    def test_plan_is_deterministic_for_a_seed_and_changes_with_it(self):
        L = self._lengths(500)
        a = fl.plan_updates(L, np.arange(500), 16, 4096, 16, np.random.default_rng(3))
        b = fl.plan_updates(L, np.arange(500), 16, 4096, 16, np.random.default_rng(3))
        c = fl.plan_updates(L, np.arange(500), 16, 4096, 16, np.random.default_rng(4))
        flat = lambda p: np.concatenate([m for u in p for m in u]).tolist()  # noqa: E731
        self.assertEqual(flat(a), flat(b))
        self.assertNotEqual(flat(a), flat(c))

    def test_sequence_longer_than_budget_gets_its_own_micro_batch(self):
        L = np.array([5000, 10, 10])
        plan = fl.plan_updates(L, np.arange(3), 16, 4096, 16, np.random.default_rng(0))
        micro = [m.tolist() for u in plan for m in u]
        self.assertIn([0], micro)

    def test_empty_plan(self):
        self.assertEqual(fl.plan_updates(np.array([1, 2]), np.array([], dtype=int), 16, 4096, 16,
                                         np.random.default_rng(0)), [])

    def test_budget_batches_cover_rows_longest_first(self):
        L = self._lengths(700)
        idx = np.arange(100, 600)
        batches = fl.budget_batches(L, idx, 16384, 64)
        seen = np.concatenate(batches)
        self.assertEqual(sorted(seen.tolist()), idx.tolist())
        firsts = [L[b[0]] for b in batches]
        self.assertEqual(firsts, sorted(firsts, reverse=True))
        for b in batches:
            self.assertLessEqual(len(b), 64)
            if len(b) > 1:
                self.assertLessEqual(len(b) * L[b].max(), 16384)


class SmallMaths(unittest.TestCase):
    def test_auc(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(fl.auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(fl.auc_score(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
        self.assertAlmostEqual(fl.auc_score(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)
        self.assertAlmostEqual(fl.auc_score(y, np.array([0.1, 0.6, 0.4, 0.9])), 0.75)
        self.assertTrue(np.isnan(fl.auc_score(np.array([1, 1]), np.array([0.1, 0.2]))))

    def test_cosine_schedule(self):
        warm, total = 10, 100
        vals = [fl.cosine_lr(u, warm, total) for u in range(total)]
        self.assertAlmostEqual(vals[0], 0.1)
        self.assertAlmostEqual(vals[warm - 1], 1.0)
        self.assertAlmostEqual(vals[warm], 1.0)
        self.assertLess(vals[-1], 0.01)
        after = vals[warm:]
        self.assertTrue(all(a >= b for a, b in zip(after, after[1:])))

    def test_grad_ckpt_flag_defaults_to_preset(self):
        base = ["--data-dir", "d", "--output", "o"]
        self.assertIsNone(fl.parse_args(base).grad_ckpt)
        self.assertTrue(fl.parse_args(base + ["--grad-ckpt"]).grad_ckpt)
        self.assertFalse(fl.parse_args(base + ["--no-grad-ckpt"]).grad_ckpt)
        self.assertEqual(fl.parse_args(base).eval_every, 0.0)
        self.assertEqual(fl.parse_args(base).max_len, 1024)


# --------------------------------------------------------------------------- run_cv
class RunCvLlm(unittest.TestCase):
    def test_llm_command_uses_the_llm_trainer_and_its_defaults(self):
        args = rc.parse_args(["--trainer", "llm", "--preset", "qwen3-8b", "--folds", "5", "--output", "r"])
        cmd = rc.build_cmd(args, Path("r"), 2)
        self.assertTrue(cmd[1].endswith("finetune_llm.py"))
        for flag in ("--max-len", "--head", "--eval-every", "--model", "--epochs", "--lr", "--overwrite"):
            self.assertNotIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--fold") + 1], "2")
        self.assertEqual(cmd[cmd.index("--preset") + 1], "qwen3-8b")

    def test_14b_command_folder_names_and_assembly_prefix(self):
        args = rc.parse_args(["--trainer", "llm", "--preset", "qwen3-14b", "--folds", "5", "--output",
                              "runs/cv-qwen3-14b", "--name", "qwen3-14b", "--eval-every", "0.5"])
        cmd = rc.build_cmd(args, Path("runs/cv-qwen3-14b"), 3)
        self.assertEqual(cmd[cmd.index("--preset") + 1], "qwen3-14b")
        self.assertEqual(Path(cmd[cmd.index("--output") + 1]).name, "qwen3-14b-f3")
        self.assertEqual(cmd[cmd.index("--eval-every") + 1], "0.5")
        self.assertEqual(rc.requested_model(args), "Qwen/Qwen3-14B-Base")
        with tempfile.TemporaryDirectory() as td:  # a hyphen and digits in the prefix must survive
            for f in range(5):
                (Path(td) / f"qwen3-14b-f{f}").mkdir()
            self.assertEqual(ac.discover_prefix(Path(td)), ("qwen3-14b", [0, 1, 2, 3, 4]))

    def test_options_are_passed_when_given(self):
        args = rc.parse_args(["--output", "r", "--max-len", "512", "--head", "128", "--eval-every", "0.5"])
        cmd = rc.build_cmd(args, Path("r"), 0)
        self.assertTrue(cmd[1].endswith("finetune_transformer.py"))
        self.assertEqual(cmd[cmd.index("--max-len") + 1], "512")
        self.assertEqual(cmd[cmd.index("--head") + 1], "128")
        self.assertEqual(cmd[cmd.index("--eval-every") + 1], "0.5")

    def test_requested_model(self):
        self.assertEqual(rc.requested_model(rc.parse_args(["--trainer", "llm", "--preset", "qwen3-8b",
                                                            "--output", "r"])), "Qwen/Qwen3-8B-Base")
        self.assertIsNone(rc.requested_model(rc.parse_args(["--output", "r"])))  # auto depends on the GPU
        self.assertEqual(rc.requested_model(rc.parse_args(["--output", "r", "--model", "x/y"])), "x/y")
        with self.assertRaises(SystemExit):
            rc.requested_model(rc.parse_args(["--trainer", "llm", "--preset", "large", "--output", "r"]))

    def _done_fold(self, root: Path, name: str, f: int, model: str):
        d = rc.fold_dir(root, name, f)
        d.mkdir(parents=True)
        for fn in ("holdout_predictions.csv", "test_predictions.csv"):
            (d / fn).write_text("id,margin\n1,0.5\n", encoding="utf-8")
        (d / "run_info.json").write_text(json.dumps({"config": {"model": model}, "total_minutes": 80.0}),
                                         encoding="utf-8")

    def test_refuses_to_mix_models_in_one_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._done_fold(root, "qwen3", 0, "Qwen/Qwen3-4B-Base")
            with self.assertRaises(SystemExit):
                rc.main(["--trainer", "llm", "--preset", "qwen3-8b", "--output", str(root),
                         "--name", "qwen3", "--dry-run"])

    def test_same_model_skips_the_finished_fold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._done_fold(root, "qwen3", 0, "Qwen/Qwen3-8B-Base")
            rc.main(["--trainer", "llm", "--preset", "qwen3-8b", "--output", str(root), "--name", "qwen3",
                     "--dry-run"])  # must not raise

    def test_force_adds_overwrite_for_finished_llm_folds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._done_fold(root, "qwen3", 1, "Qwen/Qwen3-8B-Base")
            args = rc.parse_args(["--trainer", "llm", "--output", str(root), "--name", "qwen3", "--force"])
            self.assertIn("--overwrite", rc.build_cmd(args, root, 1))
            self.assertNotIn("--overwrite", rc.build_cmd(args, root, 2))


# --------------------------------------------------------------------------- ensemble_cv
def _write_data(td: Path, n=400, nt=120, seed=0, reviews=None, test_reviews=None):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    data = td / "data"
    data.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame({"id": np.arange(1, n + 1),
                          "review": reviews if reviews is not None else ["a fine film " * 20] * n,
                          "sentiment": np.where(y == 1, "positive", "negative")})
    test = pd.DataFrame({"id": np.arange(5001, 5001 + nt),
                         "review": test_reviews if test_reviews is not None else ["some film " * 10] * nt})
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "sentiment": "positive"}).to_csv(data / "sample_submission.csv", index=False)
    return data, train, test, y


def _family(td: Path, name: str, train, test, y, acc, scale, seed):
    rng = np.random.default_rng(seed)
    good = rng.random(len(y)) < acc
    sign = np.where(good, 2 * y - 1, 1 - 2 * y)
    oof = sign * rng.gamma(3, scale, len(y))
    tm = rng.normal(0, 3 * scale, len(test))
    d = td / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": train["id"], "label": y, "margin": oof}).to_csv(d / "oof_predictions.csv", index=False)
    pd.DataFrame({"id": test["id"], "margin": tm}).to_csv(d / "test_predictions.csv", index=False)
    return d, oof, tm


class EnsembleCvCombiners(unittest.TestCase):
    def test_each_choice_applies_its_own_formula(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, [1.0, 5.0], (300, 2))
        Xn = rng.normal(0, [1.0, 5.0], (50, 2))
        y = (X.sum(1) > 0).astype(int)
        names = ["a", "b"]
        m, w = ec.combine("s-mean", X, Xn, y, names, 1e-2)
        np.testing.assert_allclose(m, (Xn / (X.std(0) + 1e-12)).mean(1))
        m, w = ec.combine("z-mean", X, Xn, y, names, 1e-2)
        np.testing.assert_allclose(m, ((Xn - X.mean(0)) / (X.std(0) + 1e-12)).mean(1))
        m, _ = ec.combine("mean", X, Xn, y, names, 1e-2)
        np.testing.assert_allclose(m, Xn.mean(1))
        m, _ = ec.combine("single:b", X, Xn, y, names, 1e-2)
        np.testing.assert_allclose(m, Xn[:, 1])
        m, _ = ec.combine("rank-mean", X, Xn, y, names, 1e-2)
        self.assertAlmostEqual(float(np.median(m)), 0.0)          # centred at the median
        self.assertLessEqual(int(np.sum(m > 0)), 25)              # at most half positive
        m, w = ec.combine("stack", X, Xn, y, names, 1e-2)
        self.assertIn("bias", w)
        with self.assertRaises(ValueError):
            ec.combine("nonsense", X, Xn, y, names, 1e-2)

    def test_forced_z_mean_writes_the_z_mean_submission(self):
        """Regression test: z-mean used to fall back silently to the single best family."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data, train, test, y = _write_data(td)
            da, oa, ta = _family(td, "fa", train, test, y, 0.95, 1.0, 1)
            db, ob, tb = _family(td, "fb", train, test, y, 0.90, 4.0, 2)
            out = td / "out"
            ec.main(["--data-dir", str(data), "--oof", f"a={da}", f"b={db}", "--output", str(out),
                     "--repeats", "2", "--choice", "z-mean"])
            X = np.column_stack([oa, ob])
            expect = ((np.column_stack([ta, tb]) - X.mean(0)) / (X.std(0) + 1e-12)).mean(1) > 0
            sub = pd.read_csv(out / "submission.csv")
            np.testing.assert_array_equal(sub["sentiment"].values == "positive", expect)
            info = json.loads((out / "ensemble_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["choice"], "z-mean")
            self.assertTrue((out / "test_margins.csv").exists())

    def test_a_family_averaged_with_itself_is_unchanged(self):
        """Regression test (found on the real NB-LR OOF, 2026-10-18): the same family listed twice must
        give exactly the single family. z-mean moved 575 of 25,000 decisions there because it re-centres."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data, train, test, y = _write_data(td)
            da, oa, ta = _family(td, "fa", train, test, y, 0.9, 1.0, 1)
            m, _ = ec.combine("s-mean", np.column_stack([oa, oa]), np.column_stack([ta, ta]), y, ["a", "b"], 1e-2)
            np.testing.assert_array_equal(m > 0, ta > 0)
            out = td / "out"
            ec.main(["--data-dir", str(data), "--oof", f"a={da}", f"b={da}", "--output", str(out), "--repeats", "2"])
            info = json.loads((out / "ensemble_info.json").read_text(encoding="utf-8"))
            self.assertTrue(info["choice"].startswith("single:"), info["choice"])  # s-mean only ties
            self.assertEqual(info["test_predictions_changed_vs_best_single"], 0)
            sub = pd.read_csv(out / "submission.csv")
            np.testing.assert_array_equal(sub["sentiment"].values == "positive", ta > 0)

    def test_automatic_choice_never_picks_a_comparison_only_combiner(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data, train, test, y = _write_data(td, n=600)
            da, _, _ = _family(td, "fa", train, test, y, 0.93, 1.0, 1)
            db, _, _ = _family(td, "fb", train, test, y, 0.93, 3.0, 2)
            out = td / "out"
            ec.main(["--data-dir", str(data), "--oof", f"a={da}", f"b={db}", "--output", str(out), "--repeats", "2"])
            info = json.loads((out / "ensemble_info.json").read_text(encoding="utf-8"))
            self.assertNotIn(info["choice"], ec.INFO_ONLY)
            self.assertIn("s-mean", info["options_estimated_accuracy"])

    def test_rejects_duplicate_names_before_reading_data(self):
        with self.assertRaises(SystemExit):
            ec.main(["--data-dir", "does-not-exist", "--oof", "a=x", "a=y", "--output", "o"])
        with self.assertRaises(SystemExit):
            ec.main(["--data-dir", "does-not-exist", "--oof", "no-equals-sign", "--output", "o"])


# --------------------------------------------------------------------------- compare_runs
class CompareRuns(unittest.TestCase):
    def test_explicit_rating(self):
        self.assertEqual(cr.explicit_rating("I give it 3/10."), "low")
        self.assertEqual(cr.explicit_rating("Solid 8 out of 10"), "high")
        self.assertEqual(cr.explicit_rating("10/10 masterpiece"), "high")
        self.assertEqual(cr.explicit_rating("the first was 9/10, this one 2/10"), "mixed")
        self.assertEqual(cr.explicit_rating("a 5/10 at best"), "mid")
        self.assertIsNone(cr.explicit_rating("no numbers here"))
        self.assertIsNone(cr.explicit_rating("released in 1999/2000"))

    def _runs(self, td: Path, seed=0):
        n, nt = 600, 150
        rng = np.random.default_rng(seed)
        reviews = [("great 9/10 " if i % 7 == 0 else "film ") * (5 + i % 60) for i in range(n)]
        test_reviews = [f"test review {i} " * 5 for i in range(nt)]
        test_reviews[3] = reviews[10]                # one test review identical to a train review
        data, train, test, y = _write_data(td, n=n, nt=nt, seed=seed, reviews=reviews, test_reviews=test_reviews)
        val = np.arange(0, n, 3)                     # both runs validated the same rows
        out = {}
        for name, acc, sc in (("A", 0.93, 2.0), ("B", 0.95, 6.0)):
            good = rng.random(len(val)) < acc
            m = np.where(good, 2 * y[val] - 1, 1 - 2 * y[val]) * rng.gamma(3, sc, len(val))
            d = td / f"run{name}"
            d.mkdir()
            pd.DataFrame({"id": train["id"].values[val], "label": y[val], "margin": m}).to_csv(
                d / "holdout_predictions.csv", index=False)
            pd.DataFrame({"id": test["id"], "margin": rng.normal(0, sc, nt)}).to_csv(
                d / "test_predictions.csv", index=False)
            out[name] = d
        return data, out

    def test_end_to_end_writes_a_complete_submission(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data, runs = self._runs(td)
            out = td / "ens"
            cr.main(["--data-dir", str(data), "--a", str(runs["A"]), "--b", str(runs["B"]),
                     "--names", "deb", "llm", "--output", str(out)])
            sub = pd.read_csv(out / "submission.csv")
            sample = pd.read_csv(data / "sample_submission.csv")
            self.assertEqual(list(sub.columns), ["id", "sentiment"])
            self.assertEqual(sub["id"].tolist(), sample["id"].tolist())
            info = json.loads((out / "compare_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["n_common"], 200)
            self.assertEqual(info["duplicates"]["test_reviews_with_identical_train_review"], 1)

    def test_the_test_ensemble_does_not_depend_on_labels(self):
        """Standardisation uses margins only, so changing every validation label must not change it."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data, runs = self._runs(td)
            cr.main(["--data-dir", str(data), "--a", str(runs["A"]), "--b", str(runs["B"]),
                     "--output", str(td / "e1")])
            tr = pd.read_csv(data / "train.csv")
            tr["sentiment"] = np.where(tr["sentiment"] == "positive", "negative", "positive")
            tr.to_csv(data / "train.csv", index=False)
            for d in runs.values():                  # keep the files consistent with the new labels
                h = pd.read_csv(d / "holdout_predictions.csv")
                h["label"] = 1 - h["label"]
                h.to_csv(d / "holdout_predictions.csv", index=False)
            cr.main(["--data-dir", str(data), "--a", str(runs["A"]), "--b", str(runs["B"]),
                     "--output", str(td / "e2")])
            s1 = pd.read_csv(td / "e1" / "submission.csv")
            s2 = pd.read_csv(td / "e2" / "submission.csv")
            pd.testing.assert_frame_equal(s1, s2)

    def test_a_model_compared_with_itself_gives_its_own_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data, runs = self._runs(td)
            out = td / "same"
            cr.main(["--data-dir", str(data), "--a", str(runs["A"]), "--b", str(runs["A"]), "--output", str(out)])
            tA = pd.read_csv(runs["A"] / "test_predictions.csv")["margin"].values
            sub = pd.read_csv(out / "submission.csv")
            np.testing.assert_array_equal(sub["sentiment"].values == "positive", tA > 0)
            info = json.loads((out / "compare_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["test_changed_vs_a"], 0)
            self.assertEqual(info["ensemble"], "s-mean")

    def test_refuses_runs_without_common_rows(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            data, runs = self._runs(td)
            h = pd.read_csv(runs["B"] / "holdout_predictions.csv")
            h["id"] = h["id"] + 100000
            h.to_csv(runs["B"] / "holdout_predictions.csv", index=False)
            with self.assertRaises(SystemExit):
                cr.main(["--data-dir", str(data), "--a", str(runs["A"]), "--b", str(runs["B"])])


if __name__ == "__main__":
    unittest.main(verbosity=2)
