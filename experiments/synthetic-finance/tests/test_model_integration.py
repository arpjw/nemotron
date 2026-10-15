"""Optional tiny native-model CPU integration. Never downloads model weights."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft"))


@unittest.skipUnless(AVAILABLE, "optional torch/transformers/peft dependencies absent")
class TinyNemotronTests(unittest.TestCase):
    def test_response_loss_adapter_gradient_and_reload(self):
        import torch
        from peft import PeftModel
        from transformers import NemotronConfig, NemotronForCausalLM
        from finmini.model import attach_trainable_adapter
        torch.manual_seed(17)
        torch.set_num_threads(1)
        config = NemotronConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                max_position_embeddings=128, partial_rotary_factor=0.5)
        base = NemotronForCausalLM(config)
        original = {k: v.clone() for k, v in base.state_dict().items()}
        model = attach_trainable_adapter(base)
        model.config.use_cache = False
        model.train()
        ids = torch.tensor([[10, 11, 12, 13, 14, 3]])
        labels = ids.clone()
        labels[:, :3] = -100
        result = model(input_ids=ids, labels=labels)
        expected = torch.nn.functional.cross_entropy(result.logits[:, :-1, :].reshape(-1, 128), labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(result.loss, expected)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable))
        optimizer = torch.optim.AdamW(trainable, lr=0.001)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            before = model(input_ids=ids).logits
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp, safe_serialization=True)
            fresh = NemotronForCausalLM(config)
            fresh.load_state_dict(original)
            restored = PeftModel.from_pretrained(fresh, tmp, is_trainable=False)
            restored.eval()
            with torch.no_grad():
                after = restored(input_ids=ids).logits
            torch.testing.assert_close(before, after)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
