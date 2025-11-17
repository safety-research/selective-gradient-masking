import unittest
import torch
import torch.nn as nn

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoConfig, GPTNeoSelfAttention

from sgtm.model.activation_masking import GPTNeoSelfAttentionActivationMasking
from sgtm.model.gradient_routing import GPTNeoSelfAttentionGradientRouting
from sgtm.model.parameter_masking import (
    GPTNeoSelfAttentionParameterMasking,
    GPTNeoSelfAttentionParameterMaskingNoProj,
)
from sgtm.model.split_linear import SplitLinearOut


class MockGPTNeoConfig(GPTNeoConfig):
    def __init__(self, retain_attn_heads=0, split_masked_weights=False):
        self.hidden_size = 16
        self.attention_layers = ["global", "local"]
        self.attention_types = [
            [["global"], "local"],
            [["global"], "local"],
        ]
        self.num_layers = 2
        self.num_heads = 4
        self.attention_dropout = 0
        self.resid_dropout = 0
        self.embed_dropout = 0
        self.max_position_embeddings = 8
        self.retain_attn_heads = retain_attn_heads
        self.split_masked_weights = split_masked_weights
        self.layer_norm_epsilon = 1e-5


class BaseGPTNeoSelfAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is BaseGPTNeoSelfAttentionTest:
            raise unittest.SkipTest("Skip BaseClass tests, it's an abstract test case")

    def setUp(self):
        torch.manual_seed(42)

        self.hidden_size = 16
        self.num_heads = 4
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_type = "global"
        self.batch_size = 2
        self.seq_len = 3

        self.input_data = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        self.target_data = torch.randn(self.batch_size, self.seq_len, self.hidden_size)

        self.model_class = None
        self.split_masked_weights = None

        self.slice_functions = {
            # Input projection weights (q, k, v) are sliced on first dimension
            "qkv_proj_weight": {
                "retain": lambda param, retain_dim: param[:retain_dim],
                "forget": lambda param, retain_dim: param[retain_dim:],
            },
            # Output projection weight is sliced on second dimension
            "out_proj_weight": {
                "retain": lambda param, retain_dim: param[:, :retain_dim],
                "forget": lambda param, retain_dim: param[:, retain_dim:],
            },
        }

        self.param_slices = {}

        # Override these in the specific test classes as needed
        self.config = {
            # Whether the forward pass should change with masking
            "forward_match": False,
            "grads": {
                True: { # data split match parameter split
                    "zero": [
                        # Parameters that should have zero gradients when masked
                    ],
                    "non_zero": [
                        # Parameters that should have non-zero gradients
                    ],
                    "match_vanilla": [
                        # Parameters whose gradients should match the vanilla model
                    ],
                    "mismatch_vanilla": [
                        # Parameters whose gradients should differ from the vanilla model
                    ],
                },
                False: { # data split mismatch parameter split
                    "zero": [
                        # Parameters that should have zero gradients when not masked
                    ],
                    "non_zero": [
                        # Parameters that should have non-zero gradients
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

    def get_slice_function(self, param_name, slice_type):
        if "out_proj.weight" in param_name:
            return self.slice_functions["out_proj_weight"][slice_type]
        elif any(x in param_name for x in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]):
            return self.slice_functions["qkv_proj_weight"][slice_type]
        else:
            raise ValueError(f"Unknown parameter type for {param_name}")

    def create_model(self, model_class, retain_heads):
        config = MockGPTNeoConfig(retain_attn_heads=retain_heads, split_masked_weights=self.split_masked_weights)
        model = model_class(config, attention_type=self.attention_type)

        torch.manual_seed(42)
        for layer in model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        return model

    def run_training_step(self, model, sgtm_mode=None):
        if sgtm_mode is not None:
            outputs = model(self.input_data, sgtm_mode=sgtm_mode)
        else:
            outputs = model(self.input_data)

        # Extract the attention output (first element of tuple)
        if isinstance(outputs, tuple):
            attn_output = outputs[0]
        else:
            attn_output = outputs

        criterion = nn.MSELoss()
        loss = criterion(attn_output, self.target_data) * 1e5
        loss.backward()

        if sgtm_mode is not None:
            model.adjust_gradients(sgtm_mode=sgtm_mode)

        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()

        # manually fill for split parameters
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

        return {
            "outputs": attn_output,
            "loss": loss.item(),
            "grads": grads,
        }

    def test_forward_shape(self):
        retain_heads = self.num_heads // 4
        model = self.create_model(self.model_class, retain_heads)

        outputs_forget = model(self.input_data, sgtm_mode="forget")
        if isinstance(outputs_forget, tuple):
            attn_output = outputs_forget[0]
        else:
            attn_output = outputs_forget
        self.assertEqual(attn_output.shape, (self.batch_size, self.seq_len, self.hidden_size))

        outputs_retain = model(self.input_data, sgtm_mode="retain")
        if isinstance(outputs_retain, tuple):
            attn_output = outputs_retain[0]
        else:
            attn_output = outputs_retain
        self.assertEqual(attn_output.shape, (self.batch_size, self.seq_len, self.hidden_size))

        outputs_default = model(self.input_data, sgtm_mode="default")
        if isinstance(outputs_default, tuple):
            attn_output = outputs_default[0]
        else:
            attn_output = outputs_default
        self.assertEqual(attn_output.shape, (self.batch_size, self.seq_len, self.hidden_size))

    def test_forward_pass(self):
        retain_heads = self.num_heads // 4

        vanilla_model = self.create_model(GPTNeoSelfAttention, None)  # Base model
        no_mask_model = self.create_model(self.model_class, 0)  # retain_heads = 0
        masked_model = self.create_model(self.model_class, retain_heads=retain_heads)  # 25% retain

        vanilla_output = vanilla_model(self.input_data)
        no_mask_output = no_mask_model(self.input_data, sgtm_mode="forget")
        forget_output = masked_model(self.input_data, sgtm_mode="forget")
        retain_output = masked_model(self.input_data, sgtm_mode="retain")
        default_output = masked_model(self.input_data, sgtm_mode="default")

        if isinstance(vanilla_output, tuple):
            vanilla_output = vanilla_output[0]
        if isinstance(no_mask_output, tuple):
            no_mask_output = no_mask_output[0]
        if isinstance(forget_output, tuple):
            forget_output = forget_output[0]
        if isinstance(retain_output, tuple):
            retain_output = retain_output[0]
        if isinstance(default_output, tuple):
            default_output = default_output[0]

        # Compare vanilla vs no masking - should always be identical
        self.assertTrue(
            torch.allclose(vanilla_output, no_mask_output, rtol=1e-4, atol=1e-4),
            "Zero retain dimension should produce same output as vanilla model",
        )

        # No-masking mode should always match vanilla output
        self.assertTrue(
            torch.allclose(vanilla_output, default_output, rtol=1e-4, atol=1e-4),
            "Non-masking mode should match vanilla model output",
        )

        if self.config["forward_match"]:
            self.assertTrue(
                torch.allclose(vanilla_output, forget_output, rtol=1e-4, atol=1e-4),
                "Masking should not affect forward pass output",
            )
        else:
            self.assertFalse(
                torch.allclose(vanilla_output, forget_output, rtol=1e-4, atol=1e-4),
                "Masking should affect forward pass output.",
            )
        self.assertFalse(
            torch.allclose(vanilla_output, retain_output, rtol=1e-4, atol=1e-4),
            "Masking should always affect forward pass output with retain data",
        )

    def test_retain_dim_effect(self):
        """Test that different retain dimensions have appropriate effects on forward pass"""
        percentages = [0.25, 0.5, 1.0]
        forget_diffs = []
        retain_diffs = []

        for pct in percentages:
            retain_heads = int(self.num_heads * pct)

            model = self.create_model(self.model_class, retain_heads)
            outputs_no_masking = model(self.input_data, sgtm_mode="default")
            forget_masking = model(self.input_data, sgtm_mode="forget")
            retain_masking = model(self.input_data, sgtm_mode="retain")

            if isinstance(outputs_no_masking, tuple):
                outputs_no_masking = outputs_no_masking[0]
            if isinstance(forget_masking, tuple):
                forget_masking = forget_masking[0]
            if isinstance(retain_masking, tuple):
                retain_masking = retain_masking[0]

            forget_diff = torch.abs(outputs_no_masking - forget_masking).mean().item()
            retain_diff = torch.abs(outputs_no_masking - retain_masking).mean().item()
            forget_diffs.append(forget_diff)
            retain_diffs.append(retain_diff)

        if self.config["forward_match"]:
            for diff in forget_diffs:
                self.assertLess(diff, 1e-4, "Masking should not affect forward pass regardless of retain dimension")
        else:
            # Check differences are monotonically increasing for both forget and retain
            for i in range(len(forget_diffs) - 1):
                self.assertLess(
                    forget_diffs[i],
                    forget_diffs[i + 1],
                    f"Masking more heads should have greater effect, "
                    f"but got {forget_diffs[i]} >= {forget_diffs[i + 1]} "
                    f"for {percentages[i] * 100}% vs {percentages[i + 1] * 100}% masked heads (forget)",
                )
                self.assertGreater(
                    retain_diffs[i],
                    retain_diffs[i + 1],
                    f"Masking more heads should have greater effect, "
                    f"but got {retain_diffs[i]} >= {retain_diffs[i + 1]} "
                    f"for {percentages[i] * 100}% vs {percentages[i + 1] * 100}% masked heads (retain)",
                )

    def test_backward_pass(self):
        retain_heads = self.num_heads // 4
        retain_dim = retain_heads * self.head_dim  # Convert to hidden dimension size

        vanilla_model = self.create_model(GPTNeoSelfAttention, None)  # Base model
        no_mask_model = self.create_model(self.model_class, 0)  # retain_heads = 0
        masking_model = self.create_model(self.model_class, retain_heads=retain_heads)  # 25% retain

        self._test_backward_pass(
            vanilla_model=vanilla_model, no_mask_model=no_mask_model, masked_model=masking_model, retain_dim=retain_dim
        )

    def test_backward_pass_sequential(self):
        retain_heads = self.num_heads // 4
        retain_dim = retain_heads * self.head_dim  # Convert to hidden dimension size
        create_model = self.create_model

        class SequentialAttention(nn.Module):
            def __init__(self, retain_heads=None, model_class=None):
                super().__init__()

                self.retain_heads = retain_heads
                self.attn1 = create_model(GPTNeoSelfAttention, None)
                if retain_heads is None:
                    self.attn2 = create_model(GPTNeoSelfAttention, None)
                else:
                    self.attn2 = create_model(model_class, retain_heads)

            def forward(self, x, sgtm_mode="default"):
                attn1_out = self.attn1(x)
                if isinstance(attn1_out, tuple):
                    x = attn1_out[0]
                else:
                    x = attn1_out

                if self.retain_heads is None:
                    attn2_out = self.attn2(x)
                else:
                    attn2_out = self.attn2(x, sgtm_mode=sgtm_mode)

                if isinstance(attn2_out, tuple):
                    return attn2_out[0]
                else:
                    return attn2_out

            def adjust_gradients(self, sgtm_mode="default"):
                if self.retain_heads is not None:
                    self.attn2.adjust_gradients(sgtm_mode=sgtm_mode)

            def ablate(self):
                if self.retain_heads is not None:
                    self.mlp2.ablate()

        vanilla_model = SequentialAttention(retain_heads=None)
        no_mask_model = SequentialAttention(retain_heads=0, model_class=self.model_class)
        masked_model = SequentialAttention(retain_heads=retain_heads, model_class=self.model_class)

        self._test_backward_pass(
            vanilla_model=vanilla_model, no_mask_model=no_mask_model, masked_model=masked_model, retain_dim=retain_dim
        )

    def test_ablate(self):
        """Test that the ablate method zeros out the appropriate weights based on implementation."""
        retain_heads = self.num_heads // 4
        retain_dim = retain_heads * self.head_dim

        model = self.create_model(self.model_class, retain_heads)

        # Store original parameters before ablation
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.clone()

        for submodule_name, submodule in model.named_modules():
            if isinstance(submodule, SplitLinearOut):
                original_params[submodule_name + ".weight"] = torch.cat(
                    [submodule.weight_retain, submodule.weight_forget], dim=0
                )
                if submodule.bias is not None:
                    original_params[submodule_name + ".bias"] = torch.cat(
                        [submodule.bias_retain, submodule.bias_forget], dim=0
                    )

        # Call ablate method
        model.ablate()

        ablated_params = {}
        for name, param in model.named_parameters():
            ablated_params[name] = param.clone()

        # TODO: move this to utils
        for submodule_name, submodule in model.named_modules():
            if isinstance(submodule, SplitLinearOut):
                ablated_params[submodule_name + ".weight"] = torch.cat(
                    [submodule.weight_retain, submodule.weight_forget], dim=0
                )
                if submodule.bias is not None:
                    ablated_params[submodule_name + ".bias"] = torch.cat(
                        [submodule.bias_retain, submodule.bias_forget], dim=0
                    )

        # Verify the ablation based on the model's configuration
        # Check zero regions (forget dimensions for parameter masking, retain for activation masking)
        zero_weights = self.config["grads"][False]["zero"]
        for param_name in zero_weights:
            if param_name in original_params:
                zero_slice_fn = self.get_slice_function(param_name, "forget")
                preserve_slice_fn = self.get_slice_function(param_name, "retain")

                zeroed_part = zero_slice_fn(ablated_params[param_name], retain_dim)
                self.assertTrue(
                    torch.allclose(zeroed_part, torch.zeros_like(zeroed_part)),
                    f"After ablation, {param_name} should have zeros in the "
                    f"{'forget' if 'ParameterMasking' in self.model_class.__name__ else 'retain'} dimensions",
                )

                preserved_part = preserve_slice_fn(ablated_params[param_name], retain_dim)
                original_preserved_part = preserve_slice_fn(original_params[param_name], retain_dim)
                self.assertTrue(
                    torch.allclose(preserved_part, original_preserved_part),
                    f"After ablation, {param_name} should preserve the "
                    f"{'retain' if 'ParameterMasking' in self.model_class.__name__ else 'forget'} dimensions",
                )

        # Run a forward pass after ablation to verify behavior
        ablated_output = model(self.input_data)
        if isinstance(ablated_output, tuple):
            ablated_output = ablated_output[0]

        self.assertEqual(
            ablated_output.shape,
            (self.batch_size, self.seq_len, self.hidden_size),
            "Output shape should be maintained after ablation",
        )

        # Test trainable ablation (random initialization instead of zeroing)
        model_trainable = self.create_model(self.model_class, retain_heads)
        # Copy original weights to ensure fair comparison
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(model.named_parameters(), model_trainable.named_parameters()):
                param2.copy_(original_params[name1])

        model_trainable.ablate(trainable=True)

        trainable_params = {}
        for name, param in model_trainable.named_parameters():
            trainable_params[name] = param.clone()

        for submodule_name, submodule in model_trainable.named_modules():
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
                zero_slice_fn = self.get_slice_function(param_name, "forget")
                preserve_slice_fn = self.get_slice_function(param_name, "retain")

                # Check that trainable ablation doesn't zero out weights
                trainable_part = zero_slice_fn(trainable_params[param_name], retain_dim)
                self.assertFalse(
                    torch.allclose(trainable_part, torch.zeros_like(trainable_part), atol=1e-6),
                    f"With trainable=True, {param_name} should have random values, not zeros",
                )

                # Check that trainable and zero ablation produce different results
                zero_part = zero_slice_fn(ablated_params[param_name], retain_dim)
                self.assertFalse(
                    torch.allclose(trainable_part, zero_part, atol=1e-6),
                    f"trainable=True and trainable=False should produce different results for {param_name}",
                )

                # Check that trainable ablation actually changed the weights
                original_part = zero_slice_fn(original_params[param_name], retain_dim)
                self.assertFalse(
                    torch.allclose(trainable_part, original_part, atol=1e-6),
                    f"With trainable=True, {param_name} should have different values from original weights",
                )

                # Check that preserved region is unchanged in trainable ablation
                preserved_part = preserve_slice_fn(trainable_params[param_name], retain_dim)
                original_preserved_part = preserve_slice_fn(original_params[param_name], retain_dim)
                self.assertTrue(
                    torch.allclose(preserved_part, original_preserved_part, atol=1e-6),
                    f"With trainable=True, preserved region of {param_name} should remain unchanged",
                )

    def _test_backward_pass(self, vanilla_model, no_mask_model, masked_model, retain_dim):
        vanilla_output = self.run_training_step(vanilla_model, sgtm_mode=None)
        no_mask_output = self.run_training_step(no_mask_model, sgtm_mode="forget")
        default_output = self.run_training_step(masked_model, sgtm_mode="default")

        # Verify vanilla matches no_mask_model weight updates
        for param_name in vanilla_output["grads"]:
            if param_name in no_mask_output["grads"]:
                self.assertTrue(
                    torch.allclose(
                        vanilla_output["grads"][param_name], no_mask_output["grads"][param_name], rtol=1e-4, atol=1e-4
                    ),
                    f"No-mask model should have same {param_name} updates as vanilla model",
                )

            if param_name in default_output["grads"]:
                self.assertTrue(
                    torch.allclose(
                        vanilla_output["grads"][param_name],
                        default_output["grads"][param_name],
                        rtol=1e-4,
                        atol=1e-4,
                    ),
                    f"No-mask forward run on the masked model should have same {param_name} updates as vanilla model",
                )

        # Test gradient patterns for both forget and retain modes
        for slice_type in ("retain", "forget"):
            for sgtm_mode in ("forget", "retain"):
                is_match = slice_type == sgtm_mode
                output = self.run_training_step(masked_model, sgtm_mode=sgtm_mode)
                slice_type_fn = slice_type
                if sgtm_mode == "retain":
                    config = ACTIVATION_MASKING_CONFIG
                else:
                    config = self.config

                for param_name in config["grads"][is_match]["zero"]:
                    if param_name not in output["grads"]:
                        continue

                    slice_fn = self.get_slice_function(param_name, slice_type_fn)
                    grad = output["grads"][param_name]
                    grad_slice = slice_fn(grad, retain_dim)

                    self.assertTrue(
                        torch.allclose(grad_slice, torch.zeros_like(grad_slice), rtol=1e-5, atol=1e-5),
                        f"Masked dimensions of {param_name} should have zero updates when sgtm_mode={sgtm_mode}",
                    )

                for param_name in config["grads"][is_match]["non_zero"]:
                    if param_name not in output["grads"]:
                        continue

                    slice_fn = self.get_slice_function(param_name, slice_type_fn)
                    grad = output["grads"][param_name]
                    grad_slice = slice_fn(grad, retain_dim)

                    # For non-zero checks we verify the norm is above a small threshold
                    self.assertGreater(
                        torch.norm(grad_slice).item(),
                        1e-5,
                        f"Non-masked dimensions of {param_name} should have non-zero updates when sgtm_mode={sgtm_mode}",
                    )

                for param_name in config["grads"][is_match]["match_vanilla"]:
                    if param_name not in output["grads"] or param_name not in vanilla_output["grads"]:
                        continue

                    slice_fn = self.get_slice_function(param_name, slice_type_fn)
                    vanilla_grad = slice_fn(vanilla_output["grads"][param_name], retain_dim)
                    masked_grad = slice_fn(output["grads"][param_name], retain_dim)

                    self.assertTrue(
                        torch.allclose(vanilla_grad, masked_grad, rtol=1e-4, atol=1e-4),
                        f"Unmasked dimensions of {param_name} should match between vanilla and masked models "
                        f"when sgtm_mode={sgtm_mode}",
                    )

                for param_name in config["grads"][is_match]["mismatch_vanilla"]:
                    if param_name not in output["grads"] or param_name not in vanilla_output["grads"]:
                        continue

                    slice_fn = self.get_slice_function(param_name, slice_type_fn)
                    vanilla_grad = slice_fn(vanilla_output["grads"][param_name], retain_dim)
                    masked_grad = slice_fn(output["grads"][param_name], retain_dim)

                    self.assertFalse(
                        torch.allclose(vanilla_grad, masked_grad, rtol=1e-4, atol=1e-4),
                        f"Masked dimensions of {param_name} should not match between vanilla and masked models "
                        f"when sgtm_mode={sgtm_mode}",
                    )


ACTIVATION_MASKING_CONFIG = {
    "forward_match": False,
    "grads": {
        False: {
            "zero": [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "out_proj.weight",
                "attn2.q_proj.weight",
                "attn2.k_proj.weight",
                "attn2.v_proj.weight",
                "attn2.out_proj.weight",
            ],
            "non_zero": [
            ],
            "match_vanilla": [
            ],
            "mismatch_vanilla": [],
        },
        True: {
            "zero": [],  
            "non_zero": [],
            "match_vanilla": [],
            "mismatch_vanilla": [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "out_proj.weight",
                "attn1.q_proj.weight",
                "attn1.k_proj.weight",
                "attn1.v_proj.weight",
                "attn1.out_proj.weight",
                "attn2.q_proj.weight",
                "attn2.k_proj.weight",
                "attn2.v_proj.weight",
                "attn2.out_proj.weight",
            ],  
        },
    },
}


class TestActivationMasking(BaseGPTNeoSelfAttentionTest):
    """Test class for the activation masking implementation"""

    def setUp(self):
        super().setUp()
        self.model_class = GPTNeoSelfAttentionActivationMasking
        self.split_masked_weights = False

        self.config.update(ACTIVATION_MASKING_CONFIG)

    def test_activation_masking_specific(self):
        """Test that masking zeroes out specific attention heads"""
        retain_heads = self.num_heads // 4

        masked_model = self.create_model(GPTNeoSelfAttentionActivationMasking, retain_heads)

        # Call the forward pass twice, once with sgtm_mode="forget" and once with sgtm_mode="default"
        outputs_forget = masked_model(self.input_data, sgtm_mode="forget")
        outputs_default = masked_model(self.input_data, sgtm_mode="default")

        # Extract attention outputs
        if isinstance(outputs_forget, tuple):
            outputs_forget = outputs_forget[0]
        if isinstance(outputs_default, tuple):
            outputs_default = outputs_default[0]

        # The outputs should be different
        self.assertFalse(
            torch.allclose(outputs_forget, outputs_default),
            "Activation masking should change the output when masking is enabled",
        )


class TestActivationMaskingSplit(TestActivationMasking):
    """Test class for the activation masking implementation"""

    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


class TestParameterMasking(BaseGPTNeoSelfAttentionTest):
    """Test class for the parameter masking implementation"""

    def setUp(self):
        super().setUp()
        self.model_class = GPTNeoSelfAttentionParameterMasking
        self.split_masked_weights = False

        # Override config for parameter masking
        self.config.update(
            {
                "forward_match": True,  # Forward pass is not affected
                "grads": {
                    False: {
                        "zero": [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "out_proj.weight",
                            "attn2.q_proj.weight",
                            "attn2.k_proj.weight",
                            "attn2.v_proj.weight",
                            "attn2.out_proj.weight",
                        ],
                        "non_zero": [],
                        "match_vanilla": [
                            "attn1.q_proj.weight",
                            "attn1.k_proj.weight",
                            "attn1.v_proj.weight",
                            "attn1.out_proj.weight",
                        ],
                        "mismatch_vanilla": [],  
                    },
                    True: {
                        "zero": [], 
                        "non_zero": [],
                        "match_vanilla": [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "out_proj.weight",
                            "attn1.q_proj.weight",
                            "attn1.k_proj.weight",
                            "attn1.v_proj.weight",
                            "attn1.out_proj.weight",
                            "attn2.q_proj.weight",
                            "attn2.k_proj.weight",
                            "attn2.v_proj.weight",
                            "attn2.out_proj.weight",
                        ],
                        "mismatch_vanilla": [], 
                    },
                },
            }
        )


class TestParameterMaskingSplit(TestParameterMasking):
    """Test class for the parameter masking implementation"""

    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


class TestParameterMaskingNoProj(BaseGPTNeoSelfAttentionTest):
    """Test class for the parameter masking implementation"""

    def setUp(self):
        super().setUp()
        self.model_class = GPTNeoSelfAttentionParameterMaskingNoProj
        self.split_masked_weights = False

        # Override config for parameter masking
        self.config.update(
            {
                "forward_match": True,  # Forward pass is not affected
                "grads": {
                    False: {
                        "zero": [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "attn2.q_proj.weight",
                            "attn2.k_proj.weight",
                            "attn2.v_proj.weight",
                        ],
                        "non_zero": [],
                        "match_vanilla": [
                            "out_proj.weight",
                            "attn2.out_proj.weight",
                            "attn1.q_proj.weight",
                            "attn1.k_proj.weight",
                            "attn1.v_proj.weight",
                            "attn1.out_proj.weight",
                        ],
                        "mismatch_vanilla": [], 
                    },
                    True: {
                        "zero": [],  
                        "non_zero": [],
                        "match_vanilla": [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "out_proj.weight",
                            "attn1.q_proj.weight",
                            "attn1.k_proj.weight",
                            "attn1.v_proj.weight",
                            "attn1.out_proj.weight",
                            "attn2.q_proj.weight",
                            "attn2.k_proj.weight",
                            "attn2.v_proj.weight",
                            "attn2.out_proj.weight",
                        ],
                        "mismatch_vanilla": [],
                    },
                },
            }
        )


class TestParameterMaskingNoProjSplit(TestParameterMaskingNoProj):
    """Test class for the parameter masking implementation"""

    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


class TestGradientRouting(BaseGPTNeoSelfAttentionTest):
    """Test class for the gradient routing implementation"""

    def setUp(self):
        super().setUp()
        self.model_class = GPTNeoSelfAttentionGradientRouting
        self.split_masked_weights = False

        # Override config for gradient routing
        self.config.update(
            {
                "forward_match": True, 
                "grads": {
                    False: {
                        "zero": [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "attn2.q_proj.weight",
                            "attn2.k_proj.weight",
                            "attn2.v_proj.weight",
                        ],
                        "non_zero": [],
                        "match_vanilla": [
                            "out_proj.weight",
                            "attn2.out_proj.weight",
                        ],
                        "mismatch_vanilla": [
                        ],
                    },
                    True: {
                        "zero": [], 
                        "non_zero": [],
                        "match_vanilla": [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "out_proj.weight",
                            "attn2.q_proj.weight",
                            "attn2.k_proj.weight",
                            "attn2.v_proj.weight",
                            "attn2.out_proj.weight",
                        ],
                        "mismatch_vanilla": [
                            "attn1.q_proj.weight",
                            "attn1.k_proj.weight",
                            "attn1.v_proj.weight",
                            "attn1.out_proj.weight",
                        ],
                    },
                },
            }
        )


class TestGradientRoutingSplit(TestGradientRouting):
    """Test class for the gradient routing implementation"""

    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


if __name__ == "__main__":
    unittest.main()
