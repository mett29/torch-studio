import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, mock_open, patch

import generate
import model
import sample
import torch


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.hparams = model.HParams(n_vocab=11, n_ctx=8, n_embd=8, n_head=2, n_layer=2)

    def test_softmax_matches_torch(self):
        x = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]])
        for dim in (0, -1):
            torch.testing.assert_close(model.softmax(x, dim), torch.softmax(x, dim))

    def test_projection_initialization_and_reuse(self):
        parameters = {}
        x = torch.randn(2, 3, 64, dtype=torch.float64)
        output = model.conv1d(x, 64, parameters=parameters)
        weights = parameters["conv1d/w"]
        self.assertEqual(weights.dtype, x.dtype)
        self.assertAlmostEqual(weights.std().item(), 0.02, delta=0.001)
        torch.testing.assert_close(output, x @ weights[0] + parameters["conv1d/b"])
        torch.testing.assert_close(output, model.conv1d(x, 64, parameters=parameters))

    def test_top_p_keeps_threshold_crossing_token_per_row(self):
        logits = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]]))
        expected = torch.tensor([[True, True, False], [False, True, True]])
        torch.testing.assert_close(
            torch.isfinite(sample.top_p_logits(logits, 0.8)), expected
        )

    def test_top_p_handles_ties_and_endpoints(self):
        logits = torch.zeros(2, 4)
        self.assertTrue(torch.equal(sample.top_p_logits(logits, 1), logits))
        self.assertTrue(
            (torch.isfinite(sample.top_p_logits(logits, 0)).sum(-1) == 1).all()
        )
        self.assertTrue(
            (torch.isfinite(sample.top_p_logits(logits, 0.5)).sum(-1) == 2).all()
        )

    def test_sampling_normalizes_over_vocabulary(self):
        # Both rows choose different tokens; normalizing across batch destroys this.
        logits = torch.tensor([[[2.0, 1.0, -1.0]], [[-1.0, 1.0, 2.0]]])
        hparams = model.HParams(n_vocab=3, n_ctx=4, n_embd=4, n_head=2, n_layer=1)
        result = {
            "logits": logits,
            "present": torch.zeros(
                model.past_shape(hparams=hparams, batch_size=2, sequence=1)
            ),
        }
        with (
            patch.object(model, "model", return_value=result),
            patch.object(
                torch, "multinomial", return_value=torch.tensor([[0], [2]])
            ) as draw,
        ):
            tokens = sample.sample_sequence(
                hparams=hparams, context=torch.tensor([[0], [1]]), length=1
            )
        torch.testing.assert_close(
            draw.call_args.args[0], torch.softmax(logits[:, -1], dim=-1)
        )
        torch.testing.assert_close(tokens, torch.tensor([[0, 0], [1, 2]]))

    def test_cached_forward_matches_full_sequence(self):
        tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        parameters = {}
        full = model.model(self.hparams, tokens, parameters=parameters)
        state = torch.random.get_rng_state().clone()
        prefix = model.model(self.hparams, tokens[:, :2], parameters=parameters)
        suffix = model.model(
            self.hparams, tokens[:, 2:], past=prefix["present"], parameters=parameters
        )
        torch.testing.assert_close(full["logits"][:, 2:], suffix["logits"])
        torch.testing.assert_close(
            full["present"], torch.cat([prefix["present"], suffix["present"]], dim=-2)
        )
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_causal_mask_prevents_future_token_influence(self):
        parameters = {}
        original = model.model(
            self.hparams, torch.tensor([[1, 2, 3]]), parameters=parameters
        )
        changed = model.model(
            self.hparams, torch.tensor([[1, 2, 9]]), parameters=parameters
        )
        torch.testing.assert_close(original["logits"][:, :2], changed["logits"][:, :2])

    def test_generation_lengths_and_context_window(self):
        context = torch.tensor([[1, 2], [2, 1]])
        with patch.object(model, "model") as forward:
            tokens = sample.sample_sequence(
                hparams=self.hparams, context=context, length=0
            )
            forward.assert_not_called()
        torch.testing.assert_close(tokens, context)
        tokens = sample.sample_sequence(
            hparams=self.hparams, context=context, length=7, top_k=1
        )
        self.assertEqual(tokens.shape, (2, 9))
        torch.testing.assert_close(tokens[:, :2], context)
        with self.assertRaises(ValueError):
            sample.sample_sequence(hparams=self.hparams, context=context, length=8)

    def test_invalid_sampling_arguments(self):
        for kwargs in (
            {"length": -1},
            {"temperature": 0},
            {"top_p": 1.1},
            {"top_k": 12},
            {"batch_size": 3},
        ):
            options = {
                "hparams": self.hparams,
                "context": torch.tensor([[1], [2]]),
                "length": 1,
            }
            options.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sample.sample_sequence(**options)

    def test_past_requires_reused_parameters(self):
        prefix = model.model(self.hparams, torch.tensor([[1]]))
        with self.assertRaises(ValueError):
            model.model(self.hparams, torch.tensor([[2]]), past=prefix["present"])

    def test_float16_attention_and_filtering_remain_finite(self):
        x = torch.randn(2, 3, 8, dtype=torch.float16)
        output, present = model.attn(x, 8, past=None, hparams=self.hparams)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(present).all())
        filtered = sample.top_k_logits(
            torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float16), 1
        )
        torch.testing.assert_close(
            torch.softmax(filtered, -1),
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float16),
        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_generation_on_mps(self):
        context = torch.tensor([[1, 2]], device="mps")
        tokens = sample.sample_sequence(
            hparams=self.hparams, context=context, length=3, top_p=0.8
        )
        self.assertEqual(tokens.device.type, "mps")
        self.assertEqual(tokens.shape, (1, 5))

    def test_generate_resamples_and_counts_partial_batches(self):
        encoder = Mock(
            encoder={"<|endoftext|>": 0}, decode=lambda tokens: str(tokens.tolist())
        )
        configurations = json.dumps(vars(self.hparams))

        def fake_sample(**kwargs):
            return torch.full((kwargs["batch_size"], 2), sampler.call_count)

        with (
            patch.object(generate.encoder, "get_encoder", return_value=encoder),
            patch("builtins.open", mock_open(read_data=configurations)),
            patch.object(sample, "sample_sequence", side_effect=fake_sample) as sampler,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            generate.sample_model(
                nsamples=5, batch_size=2, length=1, config_dir="config"
            )
        self.assertEqual(
            [call.kwargs["batch_size"] for call in sampler.call_args_list], [2, 2, 1]
        )
        parameters = sampler.call_args_list[0].kwargs["parameters"]
        self.assertTrue(
            all(
                call.kwargs["parameters"] is parameters
                for call in sampler.call_args_list
            )
        )
        text = stdout.getvalue()
        self.assertEqual(text.count(" SAMPLE "), 5)
        self.assertIn(" SAMPLE 5 ", text)
        self.assertIn("[1]", text)
        self.assertIn("[3]", text)


if __name__ == "__main__":
    unittest.main()
