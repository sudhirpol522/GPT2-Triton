import unittest

import torch

from cuda_graph import DecodeGraph
from model import GPT2, GPT2Config


class DecodeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GPT2(
            GPT2Config(vocab_size=97, block_size=16, n_layer=2, n_head=4, n_embd=32)
        ).eval()

    @torch.inference_mode()
    def test_cached_logits_match_full_forward(self):
        for batch_size, prompt_length in ((1, 1), (2, 5), (3, 13)):
            with self.subTest(batch_size=batch_size, prompt_length=prompt_length):
                tokens = torch.randint(97, (batch_size, 16), dtype=torch.int32)
                cache = self.model.create_kv_cache(batch_size)
                cache.keys.fill_(float("nan"))
                cache.values.fill_(float("nan"))
                pointers = cache.keys.data_ptr(), cache.values.data_ptr()
                expected = self.model(tokens)
                prompt_logits = self.model(tokens[:, :prompt_length], kv_cache=cache)
                torch.testing.assert_close(
                    prompt_logits, expected[:, :prompt_length], atol=1e-6, rtol=1e-5
                )
                saved_keys = cache.keys[:, :, :, :prompt_length].clone()
                saved_values = cache.values[:, :, :, :prompt_length].clone()
                for position in range(prompt_length, 16):
                    actual = self.model.decode(tokens[:, position : position + 1], cache)
                    torch.testing.assert_close(
                        actual, expected[:, position : position + 1], atol=1e-6, rtol=1e-5
                    )
                    self.assertEqual(cache.length, position + 1)
                    self.assertEqual(pointers, (cache.keys.data_ptr(), cache.values.data_ptr()))
                torch.testing.assert_close(cache.keys[:, :, :, :prompt_length], saved_keys)
                torch.testing.assert_close(cache.values[:, :, :, :prompt_length], saved_values)
                with self.assertRaisesRegex(ValueError, "capacity"):
                    self.model.decode(tokens[:, :1], cache)

    @torch.inference_mode()
    def test_greedy_generation_matches_full_forward(self):
        tokens = torch.randint(97, (2, 5))
        cache = self.model.create_kv_cache(2, max_length=10)
        logits = self.model(tokens, kv_cache=cache)
        for _ in range(5):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            expected_token = self.model(tokens)[:, -1].argmax(dim=-1, keepdim=True)
            torch.testing.assert_close(next_token, expected_token)
            tokens = torch.cat((tokens, next_token), dim=1)
            logits = self.model.decode(next_token, cache)

    @torch.inference_mode()
    def test_cache_validation(self):
        tokens = torch.ones((1, 2), dtype=torch.long)
        cache = self.model.create_kv_cache(1, max_length=3)
        with self.assertRaisesRegex(ValueError, "prompt"):
            self.model.decode(tokens[:, :1], cache)
        self.model(tokens, kv_cache=cache)
        with self.assertRaisesRegex(ValueError, "one token"):
            self.model.decode(tokens, cache)
        with self.assertRaisesRegex(ValueError, "batch"):
            self.model.decode(tokens.expand(2, -1)[:, :1], cache)
        self.model.use_triton = True
        with self.assertRaisesRegex(ValueError, "PyTorch"):
            self.model.decode(tokens[:, :1], cache)
        self.model.use_triton = False
        self.model.double()
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.model.decode(tokens[:, :1], cache)
        self.assertEqual(cache.length, 2)

    def test_cache_requires_inference(self):
        cache = self.model.create_kv_cache(1)
        with self.assertRaisesRegex(RuntimeError, "inference_mode"):
            self.model(torch.ones((1, 2), dtype=torch.long), kv_cache=cache)
        self.assertEqual(cache.length, 0)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
    )
    @torch.inference_mode()
    def test_decode_graph_advances_past_attention_tile_boundary(self):
        config = GPT2Config(vocab_size=97, block_size=72, n_layer=2, n_head=4, n_embd=64)
        model = GPT2(config).cuda().half().eval()
        tokens = torch.randint(config.vocab_size, (2, 72), device="cuda")
        expected = model(tokens)

        for use_triton in (False, True):
            with self.subTest(use_triton=use_triton):
                cache = model.create_kv_cache(batch_size=2)
                model(tokens[:, :61], kv_cache=cache)
                prefix = cache.keys[:, :, :, :61].clone()
                decoder = DecodeGraph(model, cache, use_triton_attention=use_triton)
                self.assertEqual(decoder.position.item(), 61)
                with self.assertRaisesRegex(ValueError, "one integer token"):
                    decoder.decode(tokens[:, :2])
                for position in range(61, 72):
                    actual = decoder.decode(tokens[:, position : position + 1])
                    torch.testing.assert_close(
                        actual, expected[:, position : position + 1], atol=1e-2, rtol=1e-2
                    )
                    self.assertEqual(cache.length, position + 1)
                self.assertEqual(decoder.position.item(), 72)
                torch.testing.assert_close(cache.keys[:, :, :, :61], prefix)
                with self.assertRaisesRegex(ValueError, "capacity"):
                    decoder.decode(tokens[:, :1])


if __name__ == "__main__":
    unittest.main()
