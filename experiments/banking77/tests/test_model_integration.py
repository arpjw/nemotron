import importlib.util
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft"))


@unittest.skipUnless(AVAILABLE, "optional pinned CPU dependencies absent")
class TinyNativeModelTests(unittest.TestCase):
    def test_loss_gradients_adapter_reload_and_constrained_generation(self):
        import torch
        from transformers import NemotronConfig, NemotronForCausalLM
        from peft import PeftModel
        from banking77.model import LabelTrie, attach_adapter
        torch.manual_seed(17)
        torch.set_num_threads(1)
        config = NemotronConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                partial_rotary_factor=.5, eos_token_id=3, pad_token_id=3)
        base = NemotronForCausalLM(config)
        original = {key: value.clone() for key, value in base.state_dict().items()}
        model = attach_adapter(base)
        model.config.use_cache = False
        ids = torch.tensor([[20, 21, 10, 11, 3]])
        labels = ids.clone(); labels[:, :2] = -100
        output = model(input_ids=ids, labels=labels)
        expected = torch.nn.functional.cross_entropy(output.logits[:, :-1].reshape(-1, 128), labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable))
        torch.optim.AdamW(trainable, lr=.001).step()
        model.eval()
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp, safe_serialization=True)
            fresh = NemotronForCausalLM(config); fresh.load_state_dict(original)
            restored = PeftModel.from_pretrained(fresh, temp, is_trainable=False); restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(input_ids=ids).logits, restored(input_ids=ids).logits)
            class Tokenizer:
                eos_token_id = 3
                def encode(self, label, **kwargs): return {"a": [10, 11], "b": [10, 12]}[label]
            trie = LabelTrie(Tokenizer(), ["a", "b"])
            result = restored.generate(input_ids=ids[:, :2], attention_mask=torch.ones_like(ids[:, :2]), do_sample=False,
                                       max_new_tokens=8, prefix_allowed_tokens_fn=trie.callback(2), eos_token_id=3, pad_token_id=3)
            self.assertIn(tuple(result[0, 2:].tolist()), trie.paths)
