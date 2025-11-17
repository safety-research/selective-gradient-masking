import unittest

import torch
import torch.nn as nn
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock, GPTNeoConfig, GPTNeoForCausalLM

from sgtm.model.gpt import GPTNeoBlockSGTM, GPTNeoForCausalLMSGTM
from sgtm.model.split_linear import SplitLinearOut


class MockGPTNeoConfig(GPTNeoConfig):
    def __init__(self, masking_strategy=None, masked_layers=None, split_masked_weights=False):
        super().__init__()
        self.hidden_size = 8
        self.intermediate_size = 4 * self.hidden_size
        self.num_layers = 2
        self.num_heads = 4
        self.attention_types = [
            [["global"], "local"],
            [["global"], "local"],
        ]
        self.attention_layers = ["global", "local"]
        self.max_position_embeddings = 16
        self.vocab_size = 10
        self.activation_function = "gelu"
        self.initializer_range = 0.02
        self.layer_norm_epsilon = 1e-5
        self.embed_dropout = 0
        self.attention_dropout = 0
        self.resid_dropout = 0
        self.window_size = 4

        # SGTM parameters
        self.masking_strategy = masking_strategy
        self.retain_attn_heads = 1  # 25% of heads (2/8)
        self.retain_mlp_dim = 4  # 25% of intermediate_size
        self.split_masked_weights = split_masked_weights

        if masked_layers is not None:
            self.masked_layers = masked_layers


class BaseGPTNeoSGTMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is BaseGPTNeoSGTMTest:
            raise unittest.SkipTest("Skip BaseClass tests, it's an abstract test case")

    def setUp(self):
        torch.manual_seed(42)

        self.batch_size = 2
        self.seq_len = 3
        self.vocab_size = 10
        self.hidden_size = 8
        self.num_heads = 4
        self.head_dim = self.hidden_size // self.num_heads
        self.retain_attn_dim = 1 * self.head_dim  # retain_attn_heads * head_dim
        self.retain_mlp_dim = 4

        # Input data
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        # Set the sgtm strategy
        self.masking_strategy = None  # Override in subclasses
        self.split_masked_weights = None  # Override in subclasses

        # Parameter slicing functions
        self.slice_functions = {
            # Attention input projections (q, k, v) are sliced on first dimension for heads
            "attn_qkv_weight": {
                "retain": lambda param: param[: self.retain_attn_dim],
                "forget": lambda param: param[self.retain_attn_dim :],
            },
            # Attention output projection is sliced on second dimension for heads
            "attn_out_weight": {
                "retain": lambda param: param[:, : self.retain_attn_dim],
                "forget": lambda param: param[:, self.retain_attn_dim :],
            },
            # MLP input projection
            "mlp_fc_weight": {
                "retain": lambda param: param[: self.retain_mlp_dim],
                "forget": lambda param: param[self.retain_mlp_dim :],
            },
            # MLP output projection
            "mlp_proj_weight": {
                "retain": lambda param: param[:, : self.retain_mlp_dim],
                "forget": lambda param: param[:, self.retain_mlp_dim :],
            },
        }

        # Override these in the specific test classes as needed
        self.config = {
            "forward_match": True,  # Whether forward pass output should match with/without masking
            "grads": {
                True: {
                    "zero": [
                        # Parameters that should have zero gradients when masked
                    ],
                    "non_zero": [
                        # Parameters that should have non-zero gradients when masked
                    ],
                    "match_vanilla": [
                        # Parameters whose gradients should match the vanilla model
                    ],
                    "mismatch_vanilla": [
                        # Parameters whose gradients should differ from the vanilla model
                    ],
                },
                False: {
                    "zero": [
                        # Parameters that should have zero gradients when not masked
                    ],
                    "non_zero": [
                        # Parameters that should have non-zero gradients when not masked
                    ],
                    "match_vanilla": [
                        # Parameters whose gradients should match the vanilla model
                    ],
                    "mismatch_vanilla": [
                        # Parameters whose gradients should differ from the vanilla model
                    ],
                },
            },
        }

    def get_slice_function(self, param_name):
        """Get the appropriate slicing function for a parameter"""
        if "q_proj.weight" in param_name or "k_proj.weight" in param_name or "v_proj.weight" in param_name:
            return self.slice_functions["attn_qkv_weight"]
        elif "out_proj.weight" in param_name:
            return self.slice_functions["attn_out_weight"]
        elif "c_fc.weight" in param_name:
            return self.slice_functions["mlp_fc_weight"]
        elif "c_proj.weight" in param_name:
            return self.slice_functions["mlp_proj_weight"]
        else:
            return None

    def create_models(self):
        """Create a vanilla model and a model with masking on one layer"""
        vanilla_config = MockGPTNeoConfig(masking_strategy=None, masked_layers=None)
        vanilla_model = GPTNeoForCausalLM(vanilla_config)

        # Create model with masking on the second layer only
        masking_config = MockGPTNeoConfig(
            masking_strategy=self.masking_strategy, masked_layers=[1], split_masked_weights=self.split_masked_weights
        )
        masked_model = GPTNeoForCausalLMSGTM(masking_config)

        # Merge split weights in masked model to match vanilla model structure
        masked_state_dict = masked_model.state_dict()
        keys_to_delete = []

        for module_name, module in masked_model.named_modules():
            if isinstance(module, SplitLinearOut):
                weight_name = f"{module_name}.weight"
                bias_name = f"{module_name}.bias"

                # Merge weights
                masked_state_dict[weight_name] = torch.cat([module.weight_retain, module.weight_forget], dim=0)
                keys_to_delete.extend([f"{module_name}.weight_retain", f"{module_name}.weight_forget"])

                # Merge biases if they exist
                if module.bias is not None:
                    masked_state_dict[bias_name] = torch.cat([module.bias_retain, module.bias_forget], dim=0)
                    keys_to_delete.extend([f"{module_name}.bias_retain", f"{module_name}.bias_forget"])

        # Delete old split entries
        for key in keys_to_delete:
            if key in masked_state_dict:
                del masked_state_dict[key]

        vanilla_model.load_state_dict(masked_state_dict, strict=False)

        return vanilla_model, masked_model

    def run_training_step(self, model, sgtm_mode=None):
        """Run a forward and backward pass"""
        if sgtm_mode is not None:
            outputs = model(input_ids=self.input_ids, labels=self.labels, sgtm_mode=sgtm_mode)
        else:
            outputs = model(input_ids=self.input_ids, labels=self.labels)

        loss = outputs.loss * 1e5
        loss.backward()

        if sgtm_mode is not None:
            model.adjust_gradients(sgtm_mode=sgtm_mode)

        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()

        # manually fill for split parameters
        # TODO: this should be model's method
        for submodule_name, submodule in model.named_modules():
            if isinstance(submodule, SplitLinearOut):
                grads[submodule_name + ".weight"] = torch.cat(
                    [submodule.weight_retain.grad, submodule.weight_forget.grad], dim=0
                )
                if submodule.bias is not None:
                    grads[submodule_name + ".bias"] = torch.cat(
                        [submodule.bias_retain.grad, submodule.bias_forget.grad], dim=0
                    )

        model.zero_grad()

        return {"loss": loss.item(), "logits": outputs.logits.detach().clone(), "grads": grads}

    def test_model_creation(self):
        """Test that models initialize correctly with masked layers"""
        _, masked_model = self.create_models()

        # Check that the first layer is a standard GPTNeoBlock
        self.assertIsInstance(masked_model.transformer.h[0], GPTNeoBlock)

        # Check that the second layer has masking
        self.assertIsInstance(masked_model.transformer.h[1], GPTNeoBlockSGTM)

        # Verify masked_layers is set correctly
        self.assertEqual(masked_model.transformer.masked_layers, [1])

    def test_forward_shape(self):
        """Test that model outputs have the correct shape"""
        _, masked_model = self.create_models()

        # Test with forget data
        outputs_forget = masked_model(input_ids=self.input_ids, sgtm_mode="forget")
        self.assertEqual(outputs_forget.logits.shape, (self.batch_size, self.seq_len, self.vocab_size))

        # Test with retain data
        outputs_retain = masked_model(input_ids=self.input_ids, sgtm_mode="retain")
        self.assertEqual(outputs_retain.logits.shape, (self.batch_size, self.seq_len, self.vocab_size))

        # Test with default data
        outputs_default = masked_model(input_ids=self.input_ids, sgtm_mode="default")
        self.assertEqual(outputs_default.logits.shape, (self.batch_size, self.seq_len, self.vocab_size))

    def test_forward_pass(self):
        vanilla_model, masked_model = self.create_models()

        # Run forward pass for all models
        vanilla_outputs = vanilla_model(input_ids=self.input_ids)
        forget_outputs = masked_model(input_ids=self.input_ids, sgtm_mode="forget")
        retain_outputs = masked_model(input_ids=self.input_ids, sgtm_mode="retain")
        default_outputs = masked_model(input_ids=self.input_ids, sgtm_mode="default")

        # Default mode should always match vanilla output
        self.assertTrue(
            torch.allclose(vanilla_outputs.logits, default_outputs.logits, rtol=1e-4, atol=1e-4),
            "Default mode should match vanilla model output",
        )

        if self.config["forward_match"]:
            self.assertTrue(
                torch.allclose(vanilla_outputs.logits, forget_outputs.logits, rtol=1e-4, atol=1e-4),
                "Masking should not affect forward pass output",
            )
        else:
            self.assertFalse(
                torch.allclose(vanilla_outputs.logits, forget_outputs.logits, rtol=1e-4, atol=1e-4),
                "Masking should affect forward pass output",
            )
        self.assertFalse(
            torch.allclose(vanilla_outputs.logits, retain_outputs.logits, rtol=1e-4, atol=1e-4),
            "Masking should always affect forward pass output with retain data",
        )

    def test_backward_pass(self):
        vanilla_model, masked_model = self.create_models()

        # Run forward and backward pass for all models
        vanilla_result = self.run_training_step(vanilla_model)
        default_result = self.run_training_step(masked_model, sgtm_mode="default")

        # Verify default mode matches vanilla model gradients
        for param_name in vanilla_result["grads"]:
            if param_name in default_result["grads"]:
                self.assertTrue(
                    torch.allclose(
                        vanilla_result["grads"][param_name],
                        default_result["grads"][param_name],
                        rtol=1e-4,
                        atol=1e-4,
                    ),
                    f"Default mode should have same {param_name} updates as vanilla model",
                )

        # Check gradients based on config for both forget and retain modes
        for slice_type in ("retain", "forget"):
            for sgtm_mode in ("forget", "retain"):
                is_match = slice_type == sgtm_mode
                result = self.run_training_step(masked_model, sgtm_mode=sgtm_mode)
                if sgtm_mode == "retain":
                    config = ACTIVATION_MASKING_CONFIG
                else:
                    config = self.config

                for param_name in config["grads"][is_match]["zero"]:
                    if param_name not in result["grads"]:
                        raise ValueError(f"Parameter {param_name} not found in masked model gradients")

                    slice_fn = self.get_slice_function(param_name)[slice_type]
                    grad = result["grads"][param_name]
                    grad_slice = slice_fn(grad)

                    self.assertTrue(
                        torch.allclose(grad_slice, torch.zeros_like(grad_slice), rtol=1e-5, atol=1e-5),
                        f"Masked dimensions of {param_name} should have zero updates when sgtm_mode={sgtm_mode}",
                    )

                for param_name in config["grads"][is_match]["non_zero"]:
                    if param_name not in result["grads"]:
                        raise ValueError(f"Parameter {param_name} not found in masked model gradients")

                    slice_fn = self.get_slice_function(param_name)[slice_type]
                    grad = result["grads"][param_name]
                    grad_slice = slice_fn(grad)

                    self.assertTrue(
                        torch.all(torch.abs(grad_slice) > 0),
                        f"Non-masked dimensions of {param_name} should all be non-zero when sgtm_mode={sgtm_mode}",
                    )

                for param_name in config["grads"][is_match]["match_vanilla"]:
                    if param_name not in result["grads"]:
                        raise ValueError(f"Parameter {param_name} not found in masked model gradients")

                    slice_fn = self.get_slice_function(param_name)[slice_type]
                    vanilla_grad = slice_fn(vanilla_result["grads"][param_name])
                    masked_grad = slice_fn(result["grads"][param_name])

                    self.assertTrue(
                        torch.allclose(vanilla_grad, masked_grad, rtol=1e-4, atol=1e-4),
                        f"Unmasked dimensions of {param_name} should match between vanilla and masked models "
                        f"when sgtm_mode={sgtm_mode}",
                    )

                for param_name in config["grads"][is_match]["mismatch_vanilla"]:
                    if param_name not in result["grads"]:
                        raise ValueError(f"Parameter {param_name} not found in masked model gradients")

                    slice_fn = self.get_slice_function(param_name)[slice_type]
                    vanilla_grad = slice_fn(vanilla_result["grads"][param_name])
                    masked_grad = slice_fn(result["grads"][param_name])

                    self.assertFalse(
                        torch.allclose(vanilla_grad, masked_grad, rtol=1e-4, atol=1e-4),
                        f"Masked dimensions of {param_name} should not match between vanilla and masked models "
                        f"when sgtm_mode={sgtm_mode}",
                    )

    def test_named_parameters(self):
        if not self.split_masked_weights:
            self.skipTest("Test only applies to models with split weights")

        vanilla_model, masked_model = self.create_models()

        all_params_vanilla = set([x[0] for x in vanilla_model.named_parameters()])
        all_params_masked = set([x[0] for x in masked_model.named_parameters()])

        common_params = all_params_vanilla & all_params_masked

        self.assertGreater(len(common_params), 0, "There should be parameters common between vanilla and masked models")

        forget_params_expected = [
            "transformer.h.1.attn.attention.k_proj.weight_forget",
            "transformer.h.1.attn.attention.v_proj.weight_forget",
            "transformer.h.1.attn.attention.q_proj.weight_forget",
            "transformer.h.1.mlp.c_fc.weight_forget",
            "transformer.h.1.mlp.c_fc.bias_forget",
        ]

        retain_params_expected = [
            "transformer.h.1.attn.attention.k_proj.weight_retain",
            "transformer.h.1.attn.attention.v_proj.weight_retain",
            "transformer.h.1.attn.attention.q_proj.weight_retain",
            "transformer.h.1.mlp.c_fc.weight_retain",
            "transformer.h.1.mlp.c_fc.bias_retain",
        ]

        forget_params = set([x[0] for x in masked_model.named_parameters_split(sgtm_split="forget")])
        retain_params = set([x[0] for x in masked_model.named_parameters_split(sgtm_split="retain")])
        joint_params = set([x[0] for x in masked_model.named_parameters_split(sgtm_split="joint")])

        for param_name in common_params:
            self.assertIn(param_name, joint_params, f"Parameter {param_name} should be in retain parameters")

        for param_name in retain_params_expected:
            self.assertIn(param_name, retain_params, f"Parameter {param_name} should be in retain parameters")

        for param_name in forget_params_expected:
            self.assertIn(param_name, forget_params, f"Parameter {param_name} should be in forget parameters")

        self.assertEqual(
            len(forget_params), len(forget_params_expected), "Number of forget parameters does not match expected"
        )

        self.assertEqual(
            len(retain_params) + len(forget_params) + len(joint_params),
            len(retain_params_expected) + len(forget_params_expected) + len(common_params) + 1, # 1 for lm_head.bias
            "Number of retain parameters does not match expected",
        )

    def test_ablate(self):
        """Test that the ablate method works correctly at the model level."""
        _, masked_model = self.create_models()

        # Store original parameters before ablation
        original_params = {}
        for name, param in masked_model.named_parameters():
            original_params[name] = param.clone()

        for submodule_name, submodule in masked_model.named_modules():
            if isinstance(submodule, SplitLinearOut):
                original_params[submodule_name + ".weight"] = torch.cat(
                    [submodule.weight_retain, submodule.weight_forget], dim=0
                )
                if submodule.bias is not None:
                    original_params[submodule_name + ".bias"] = torch.cat(
                        [submodule.bias_retain, submodule.bias_forget], dim=0
                    )

        # Test standard ablation (trainable=False)
        masked_model.ablate(trainable=False)

        ablated_params = {}
        for name, param in masked_model.named_parameters():
            ablated_params[name] = param.clone()

        for submodule_name, submodule in masked_model.named_modules():
            if isinstance(submodule, SplitLinearOut):
                ablated_params[submodule_name + ".weight"] = torch.cat(
                    [submodule.weight_retain, submodule.weight_forget], dim=0
                )
                if submodule.bias is not None:
                    ablated_params[submodule_name + ".bias"] = torch.cat(
                        [submodule.bias_retain, submodule.bias_forget], dim=0
                    )

        # Verify ablation based on masking strategy
        zero_weights = self.config["grads"][False]["zero"]
        for param_name in zero_weights:
            if param_name in original_params:
                slice_fn = self.get_slice_function(param_name)
                if slice_fn is not None:
                    # Check that the forget part was zeroed
                    zeroed_part = slice_fn["forget"](ablated_params[param_name])
                    self.assertTrue(
                        torch.allclose(zeroed_part, torch.zeros_like(zeroed_part), atol=1e-6),
                        f"After ablation, forget dimensions of {param_name} should be zero",
                    )

                    # Check that the retain part was preserved
                    preserved_part = slice_fn["retain"](ablated_params[param_name])
                    original_preserved_part = slice_fn["retain"](original_params[param_name])
                    self.assertTrue(
                        torch.allclose(preserved_part, original_preserved_part, atol=1e-6),
                        f"After ablation, retain dimensions of {param_name} should be preserved",
                    )

        # Test trainable ablation
        _, masked_model_trainable = self.create_models()
        # Copy original weights
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                masked_model.named_parameters(), masked_model_trainable.named_parameters()
            ):
                param2.copy_(original_params[name1])

        masked_model_trainable.ablate(trainable=True)

        trainable_params = {}
        for name, param in masked_model_trainable.named_parameters():
            trainable_params[name] = param.clone()

        for submodule_name, submodule in masked_model_trainable.named_modules():
            if isinstance(submodule, SplitLinearOut):
                trainable_params[submodule_name + ".weight"] = torch.cat(
                    [submodule.weight_retain, submodule.weight_forget], dim=0
                )
                if submodule.bias is not None:
                    trainable_params[submodule_name + ".bias"] = torch.cat(
                        [submodule.bias_retain, submodule.bias_forget], dim=0
                    )

        # Verify trainable ablation behavior
        for param_name in zero_weights:
            if param_name in original_params:
                slice_fn = self.get_slice_function(param_name)
                if slice_fn is not None:
                    # Check that trainable ablation doesn't zero out weights
                    trainable_part = slice_fn["forget"](trainable_params[param_name])
                    self.assertFalse(
                        torch.allclose(trainable_part, torch.zeros_like(trainable_part), atol=1e-6),
                        f"With trainable=True, forget dimensions of {param_name} should not be zero",
                    )

                    # Check that trainable and zero ablation produce different results
                    zero_part = slice_fn["forget"](ablated_params[param_name])
                    self.assertFalse(
                        torch.allclose(trainable_part, zero_part, atol=1e-6),
                        f"trainable=True and trainable=False should produce different results for {param_name}",
                    )

                    # Check that trainable ablation actually changed the weights
                    original_part = slice_fn["forget"](original_params[param_name])
                    self.assertFalse(
                        torch.allclose(trainable_part, original_part, atol=1e-6),
                        f"With trainable=True, {param_name} should have different values from original weights",
                    )

                    # Check that preserved region is unchanged in trainable ablation
                    preserved_part = slice_fn["retain"](trainable_params[param_name])
                    original_preserved_part = slice_fn["retain"](original_params[param_name])
                    self.assertTrue(
                        torch.allclose(preserved_part, original_preserved_part, atol=1e-6),
                        f"With trainable=True, retain dimensions of {param_name} should remain unchanged",
                    )

        # Run forward pass to ensure model still works after ablation
        output = masked_model(input_ids=self.input_ids, sgtm_mode="forget")
        self.assertEqual(
            output.logits.shape,
            (self.batch_size, self.seq_len, self.vocab_size),
            "Output shape should be maintained after ablation",
        )

        trainable_output = masked_model_trainable(input_ids=self.input_ids, sgtm_mode="forget")
        self.assertEqual(
            trainable_output.logits.shape,
            (self.batch_size, self.seq_len, self.vocab_size),
            "Output shape should be maintained after trainable ablation",
        )


class TestGradientRouting(BaseGPTNeoSGTMTest):
    def setUp(self):
        super().setUp()
        self.masking_strategy = "gradient_routing"
        self.split_masked_weights = False

        self.config.update(
            {
                "forward_match": True,
                "grads": {
                    False: {
                        "zero": [
                            "transformer.h.1.attn.attention.q_proj.weight",
                            "transformer.h.1.attn.attention.k_proj.weight",
                            "transformer.h.1.attn.attention.v_proj.weight",
                            "transformer.h.1.mlp.c_fc.weight",
                        ],
                        "non_zero": [
                            "transformer.h.1.attn.attention.out_proj.weight",
                        ],
                        "match_vanilla": [
                            "transformer.h.1.mlp.c_proj.weight",
                        ],
                        "mismatch_vanilla": [],
                    },
                    True: {
                        "zero": [],
                        "non_zero": [
                            "transformer.h.1.attn.attention.q_proj.weight",
                            "transformer.h.1.attn.attention.k_proj.weight",
                            "transformer.h.1.attn.attention.v_proj.weight",
                            "transformer.h.1.attn.attention.out_proj.weight",
                            "transformer.h.1.mlp.c_fc.weight",
                        ],
                        "match_vanilla": [
                            "transformer.h.1.mlp.c_proj.weight",
                        ],
                        "mismatch_vanilla": [
                            "transformer.h.0.attn.attention.out_proj.weight",
                            "transformer.h.0.attn.attention.q_proj.weight",
                            "transformer.h.0.attn.attention.k_proj.weight",
                            "transformer.h.0.attn.attention.v_proj.weight",
                            "transformer.h.0.mlp.c_fc.weight",
                        ],
                    },
                },
            }
        )


class TestGradientRoutingSplit(TestGradientRouting):
    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


ACTIVATION_MASKING_CONFIG = {
    "forward_match": False,  # Forward pass is affected
    "grads": {
        False: {
            "zero": [
                # Input and output projections have zero gradients when masked
                "transformer.h.1.attn.attention.out_proj.weight",
                "transformer.h.1.attn.attention.q_proj.weight",
                "transformer.h.1.attn.attention.k_proj.weight",
                "transformer.h.1.attn.attention.v_proj.weight",
                "transformer.h.1.mlp.c_fc.weight",
                "transformer.h.1.mlp.c_proj.weight",
            ],
            "non_zero": [],
            "match_vanilla": [],
            "mismatch_vanilla": [
                # All gradients should be affected since activations are masked
            ],
        },
        True: {
            "zero": [],  # No zero gradients when not masked
            "non_zero": [],
            "match_vanilla": [],
            "mismatch_vanilla": [
                "transformer.h.0.attn.attention.out_proj.weight",
                "transformer.h.0.attn.attention.q_proj.weight",
                "transformer.h.0.attn.attention.k_proj.weight",
                "transformer.h.0.attn.attention.v_proj.weight",
                "transformer.h.0.mlp.c_fc.weight",
                "transformer.h.0.mlp.c_proj.weight",
                "transformer.h.1.attn.attention.out_proj.weight",
                "transformer.h.1.attn.attention.q_proj.weight",
                "transformer.h.1.attn.attention.k_proj.weight",
                "transformer.h.1.attn.attention.v_proj.weight",
                "transformer.h.1.mlp.c_fc.weight",
                "transformer.h.1.mlp.c_proj.weight",
            ],
        },
    },
}


class TestActivationMasking(BaseGPTNeoSGTMTest):
    def setUp(self):
        super().setUp()
        self.masking_strategy = "activation_masking"
        self.split_masked_weights = False

        self.config.update(
            {
                "forward_match": False,  # Forward pass is affected
                "grads": {
                    False: {
                        "zero": [
                            # Input and output projections have zero gradients when masked
                            "transformer.h.1.attn.attention.out_proj.weight",
                            "transformer.h.1.attn.attention.q_proj.weight",
                            "transformer.h.1.attn.attention.k_proj.weight",
                            "transformer.h.1.attn.attention.v_proj.weight",
                            "transformer.h.1.mlp.c_fc.weight",
                            "transformer.h.1.mlp.c_proj.weight",
                        ],
                        "non_zero": [],
                        "match_vanilla": [],
                        "mismatch_vanilla": [
                            # All gradients should be affected since activations are masked
                        ],
                    },
                    True: {
                        "zero": [],  
                        "non_zero": [],
                        "match_vanilla": [],
                        "mismatch_vanilla": [
                            "transformer.h.0.attn.attention.out_proj.weight",
                            "transformer.h.0.attn.attention.q_proj.weight",
                            "transformer.h.0.attn.attention.k_proj.weight",
                            "transformer.h.0.attn.attention.v_proj.weight",
                            "transformer.h.0.mlp.c_fc.weight",
                            "transformer.h.0.mlp.c_proj.weight",
                            "transformer.h.1.attn.attention.out_proj.weight",
                            "transformer.h.1.attn.attention.q_proj.weight",
                            "transformer.h.1.attn.attention.k_proj.weight",
                            "transformer.h.1.attn.attention.v_proj.weight",
                            "transformer.h.1.mlp.c_fc.weight",
                            "transformer.h.1.mlp.c_proj.weight",
                        ],
                    },
                },
            }
        )


class TestActivationMaskingSplit(TestActivationMasking):
    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


class TestParameterMasking(BaseGPTNeoSGTMTest):
    def setUp(self):
        super().setUp()
        self.masking_strategy = "parameter_masking"
        self.split_masked_weights = False

        self.config.update(
            {
                "forward_match": True,  # Forward pass is not affected
                "grads": {
                    False: {
                        "zero": [
                            "transformer.h.1.attn.attention.out_proj.weight",
                            "transformer.h.1.attn.attention.q_proj.weight",
                            "transformer.h.1.attn.attention.k_proj.weight",
                            "transformer.h.1.attn.attention.v_proj.weight",
                            "transformer.h.1.mlp.c_fc.weight",
                            "transformer.h.1.mlp.c_proj.weight",
                        ],
                        "non_zero": [],
                        "match_vanilla": [
                            "transformer.h.0.attn.attention.out_proj.weight",
                            "transformer.h.0.attn.attention.q_proj.weight",
                            "transformer.h.0.attn.attention.k_proj.weight",
                            "transformer.h.0.attn.attention.v_proj.weight",
                            "transformer.h.0.mlp.c_fc.weight",
                            "transformer.h.0.mlp.c_proj.weight",
                        ],
                        "mismatch_vanilla": [],
                    },
                    True: {
                        "zero": [], 
                        "non_zero": [],
                        "match_vanilla": [
                            "transformer.h.0.attn.attention.out_proj.weight",
                            "transformer.h.0.attn.attention.q_proj.weight",
                            "transformer.h.0.attn.attention.k_proj.weight",
                            "transformer.h.0.attn.attention.v_proj.weight",
                            "transformer.h.0.mlp.c_fc.weight",
                            "transformer.h.0.mlp.c_proj.weight",
                            "transformer.h.1.attn.attention.out_proj.weight",
                            "transformer.h.1.attn.attention.q_proj.weight",
                            "transformer.h.1.attn.attention.k_proj.weight",
                            "transformer.h.1.attn.attention.v_proj.weight",
                            "transformer.h.1.mlp.c_fc.weight",
                            "transformer.h.1.mlp.c_proj.weight",
                        ],
                        "mismatch_vanilla": [],
                    },
                },
            }
        )


class TestParameterMaskingSplit(TestParameterMasking):
    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


class TestParameterMaskingNoProj(BaseGPTNeoSGTMTest):
    def setUp(self):
        super().setUp()
        self.masking_strategy = "parameter_masking_no_proj"
        self.split_masked_weights = False

        self.config.update(
            {
                "forward_match": True,  
                "grads": {
                    False: {
                        "zero": [
                            "transformer.h.1.attn.attention.q_proj.weight",
                            "transformer.h.1.attn.attention.k_proj.weight",
                            "transformer.h.1.attn.attention.v_proj.weight",
                            "transformer.h.1.mlp.c_fc.weight",
                        ],
                        "non_zero": [],
                        "match_vanilla": [
                            "transformer.h.1.attn.attention.out_proj.weight",
                            "transformer.h.1.mlp.c_proj.weight",
                            "transformer.h.0.attn.attention.out_proj.weight",
                            "transformer.h.0.attn.attention.q_proj.weight",
                            "transformer.h.0.attn.attention.k_proj.weight",
                            "transformer.h.0.attn.attention.v_proj.weight",
                            "transformer.h.0.mlp.c_fc.weight",
                            "transformer.h.0.mlp.c_proj.weight",
                        ],
                        "mismatch_vanilla": [],
                    },
                    True: {
                        "zero": [],  
                        "non_zero": [],
                        "match_vanilla": [
                            "transformer.h.0.attn.attention.out_proj.weight",
                            "transformer.h.0.attn.attention.q_proj.weight",
                            "transformer.h.0.attn.attention.k_proj.weight",
                            "transformer.h.0.attn.attention.v_proj.weight",
                            "transformer.h.0.mlp.c_fc.weight",
                            "transformer.h.0.mlp.c_proj.weight",
                            "transformer.h.1.attn.attention.out_proj.weight",
                            "transformer.h.1.attn.attention.q_proj.weight",
                            "transformer.h.1.attn.attention.k_proj.weight",
                            "transformer.h.1.attn.attention.v_proj.weight",
                            "transformer.h.1.mlp.c_fc.weight",
                            "transformer.h.1.mlp.c_proj.weight",
                        ],
                        "mismatch_vanilla": [],
                    },
                },
            }
        )


class TestParameterMaskingNoProjSplit(TestParameterMaskingNoProj):
    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


if __name__ == "__main__":
    unittest.main()
