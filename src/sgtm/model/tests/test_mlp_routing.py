import unittest
import torch
import torch.nn as nn

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoConfig, GPTNeoMLP

from sgtm.model.activation_masking import GPTNeoMLPActivationMasking
from sgtm.model.gradient_routing import GPTNeoMLPGradientRouting
from sgtm.model.parameter_masking import (
    GPTNeoMLPParameterMasking,
    GPTNeoMLPParameterMaskingNoProj,
)
from sgtm.model.split_linear import SplitLinearOut


class MockGPTNeoConfig(GPTNeoConfig):
    """Mock configuration that has required parameters for GPTNeoMLP"""

    def __init__(self, retain_mlp_dim=0, split_masked_weights=False):
        self.hidden_size = 4
        self.activation_function = "gelu"
        self.embed_dropout = 0
        self.attention_dropout = 0
        self.resid_dropout = 0
        self.layer_norm_epsilon = 1e-5
        self.retain_mlp_dim = retain_mlp_dim
        self.split_masked_weights = split_masked_weights


class BaseGPTNeoMLPTest(unittest.TestCase):
    """Base test class for all strategies.
    This is an abstract class that should not be run directly."""

    @classmethod
    def setUpClass(cls):
        if cls is BaseGPTNeoMLPTest:
            raise unittest.SkipTest("Skip BaseClass tests, it's an abstract test case")

    def setUp(self):
        torch.manual_seed(42)

        self.intermediate_size = 16
        self.hidden_size = 4
        self.batch_size = 2
        self.seq_len = 3
        self.lr = 1.0

        self.input_data = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        self.target_data = torch.randn(self.batch_size, self.seq_len, self.hidden_size)

        self.model_class = None
        self.split_masked_weights = None

        self.slice_functions = {
            "fc_weight": {
                "retain": lambda param, retain_dim: param[:retain_dim],
                "forget": lambda param, retain_dim: param[retain_dim:],
            },
            "fc_bias": {
                "retain": lambda param, retain_dim: param[:retain_dim],
                "forget": lambda param, retain_dim: param[retain_dim:],
            },
            "proj_weight": {
                "retain": lambda param, retain_dim: param[:, :retain_dim],
                "forget": lambda param, retain_dim: param[:, retain_dim:],
            },
        }

        self.param_slices = {}

        # Override these in the specific test classes as needed
        self.config = {
            "forward_match": False,
            "grads": {
                True: {
                    "zero": [
                        # Parameters that should have zero gradients when masked
                    ],
                    "non_zero": [
                        # Parameters that should have zero gradients when masked
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
                        # Parameters that should have zero gradients when masked
                    ],
                    "non_zero": [
                        # Parameters that should have zero gradients when masked
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
        if param_name.endswith("c_fc.weight"):
            return self.slice_functions["fc_weight"][slice_type]
        elif param_name.endswith("c_fc.bias"):
            return self.slice_functions["fc_bias"][slice_type]
        elif param_name.endswith("c_proj.weight"):
            return self.slice_functions["proj_weight"][slice_type]
        else:
            raise ValueError(f"Unknown parameter type for {param_name}")

    def create_model(self, model_class, retain_dim):
        config = MockGPTNeoConfig(retain_mlp_dim=retain_dim, split_masked_weights=self.split_masked_weights)
        model = model_class(self.intermediate_size, config)

        torch.manual_seed(42)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        return model

    def run_training_step(self, model, sgtm_mode=None):
        if sgtm_mode is not None:
            outputs = model(self.input_data, sgtm_mode=sgtm_mode)
        else:
            outputs = model(self.input_data)

        criterion = nn.MSELoss()
        loss = criterion(outputs, self.target_data) * 1e5
        loss.backward()

        if sgtm_mode is not None:
            model.adjust_gradients(sgtm_mode=sgtm_mode)

        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad

        # manually fill for split parameters
        for submodule_name, submodule in model.named_modules():
            if isinstance(submodule, SplitLinearOut):
                grads[submodule_name + ".weight"] = torch.cat(
                    [submodule.weight_retain.grad, submodule.weight_forget.grad], dim=0
                )
                grads[submodule_name + ".bias"] = torch.cat(
                    [submodule.bias_retain.grad, submodule.bias_forget.grad], dim=0
                )

        model.zero_grad()

        return {
            "outputs": outputs,
            "loss": loss.item(),
            "grads": grads,
        }

    def test_forward_shape(self):
        retain_dim = self.intermediate_size // 4
        model = self.create_model(self.model_class, retain_dim)

        outputs_forget = model(self.input_data, sgtm_mode="forget")
        self.assertEqual(outputs_forget.shape, (self.batch_size, self.seq_len, self.hidden_size))

        outputs_default = model(self.input_data, sgtm_mode="default")
        self.assertEqual(outputs_default.shape, (self.batch_size, self.seq_len, self.hidden_size))

    def test_forward_pass(self):
        retain_dim = self.intermediate_size // 4

        vanilla_model = self.create_model(GPTNeoMLP, None)  # Base model
        no_mask_model = self.create_model(self.model_class, 0)  # retain_dim = 0
        masked_model = self.create_model(self.model_class, retain_dim=retain_dim)  # 25% retain

        vanilla_output = vanilla_model(self.input_data)
        no_mask_output = no_mask_model(self.input_data, sgtm_mode="forget")
        forget_output = masked_model(self.input_data, sgtm_mode="forget")
        retain_output = masked_model(self.input_data, sgtm_mode="retain")
        default_output = masked_model(self.input_data, sgtm_mode="default")

        # Compare vanilla vs no mask - should always be identical
        self.assertTrue(
            torch.allclose(vanilla_output, no_mask_output, rtol=1e-4, atol=1e-4),
            "Zero retain dimension should produce same output as vanilla model",
        )

        # Default mode should always match vanilla output
        self.assertTrue(
            torch.allclose(vanilla_output, default_output, rtol=1e-4, atol=1e-4),
            "Default mode should match vanilla model output",
        )

        if self.config["forward_match"]:
            self.assertTrue(
                torch.allclose(vanilla_output, forget_output, rtol=1e-4, atol=1e-4),
                "Masking should not affect forward pass output",
            )
        else:
            self.assertFalse(
                torch.allclose(vanilla_output, forget_output, rtol=1e-4, atol=1e-4),
                "Masking should affect forward pass output",
            )

        self.assertFalse(
            torch.allclose(vanilla_output, retain_output, rtol=1e-4, atol=1e-4),
            "Masking should always affect forward pass output",
        )

    def test_retain_dim_effect(self):
        percentages = [0.25, 0.5, 1.0]
        forget_diffs = []
        retain_diffs = []

        for pct in percentages:
            retain_dim = int(self.intermediate_size * pct)

            model = self.create_model(self.model_class, retain_dim)
            outputs_default = model(self.input_data, sgtm_mode="default")
            output_forget = model(self.input_data, sgtm_mode="forget")
            output_retain = model(self.input_data, sgtm_mode="retain")

            forget_diff = torch.abs(outputs_default - output_forget).mean().item()
            retain_diff = torch.abs(outputs_default - output_retain).mean().item()
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
                    f"Masking more dimensions should have greater effect, "
                    f"but got {forget_diffs[i]} >= {forget_diffs[i + 1]} "
                    f"for {percentages[i] * 100}% vs {percentages[i + 1] * 100}% masked dimensions (forget)",
                )
                self.assertGreater(
                    retain_diffs[i],
                    retain_diffs[i + 1],
                    f"Masking more dimensions should have greater effect, "
                    f"but got {retain_diffs[i]} >= {retain_diffs[i + 1]} "
                    f"for {percentages[i] * 100}% vs {percentages[i + 1] * 100}% masked dimensions (retain)",
                )

    def test_backward_pass(self):
        retain_dim = self.intermediate_size // 4

        vanilla_model = self.create_model(GPTNeoMLP, None)  # Base model
        no_mask_model = self.create_model(self.model_class, 0)  # retain_dim = 0
        masked_model = self.create_model(self.model_class, retain_dim=retain_dim)  # 25% retain

        self._test_backward_pass(
            vanilla_model=vanilla_model, no_mask_model=no_mask_model, masked_model=masked_model, retain_dim=retain_dim
        )

    def test_backward_pass_sequential(self):
        retain_dim = self.intermediate_size // 4
        create_model = self.create_model

        class SequentialMLP(nn.Module):
            def __init__(self, retain_dim=None, model_class=None):
                super().__init__()

                self.retain_dim = retain_dim
                self.mlp1 = create_model(GPTNeoMLP, None)
                if retain_dim is None:
                    self.mlp2 = create_model(GPTNeoMLP, None)
                else:
                    self.mlp2 = create_model(model_class, retain_dim)

            def forward(self, x, sgtm_mode="default"):
                x = self.mlp1(x)
                if self.retain_dim is None:
                    x = self.mlp2(x)
                else:
                    x = self.mlp2(x, sgtm_mode=sgtm_mode)
                return x

            def adjust_gradients(self, sgtm_mode="default"):
                if self.retain_dim is not None:
                    self.mlp2.adjust_gradients(sgtm_mode=sgtm_mode)

            def ablate(self):
                if self.retain_dim is not None:
                    self.mlp2.ablate()

        vanilla_model = SequentialMLP(retain_dim=None)
        no_mask_model = SequentialMLP(retain_dim=0, model_class=self.model_class)
        masked_model = SequentialMLP(retain_dim=retain_dim, model_class=self.model_class)

        self._test_backward_pass(
            vanilla_model=vanilla_model, no_mask_model=no_mask_model, masked_model=masked_model, retain_dim=retain_dim
        )

    def test_ablate(self):
        """Test that the ablate method zeros out the appropriate weights based on implementation."""
        retain_dim = self.intermediate_size // 4
        model = self.create_model(self.model_class, retain_dim)

        # Store original parameters before ablation
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.clone()

        for submodule_name, submodule in model.named_modules():
            if isinstance(submodule, SplitLinearOut):
                original_params[submodule_name + ".weight"] = torch.cat(
                    [submodule.weight_retain, submodule.weight_forget], dim=0
                )
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
                ablated_params[submodule_name + ".bias"] = torch.cat(
                    [submodule.bias_retain, submodule.bias_forget], dim=0
                )

        # Verify the ablation based on the model's configuration
        # Check zero regions
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
        self.assertEqual(
            ablated_output.shape,
            (self.batch_size, self.seq_len, self.hidden_size),
            "Output shape should be maintained after ablation",
        )

        # Test trainable ablation (random initialization instead of zeroing)
        model_trainable = self.create_model(self.model_class, retain_dim)
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
            self.assertTrue(
                torch.allclose(
                    vanilla_output["grads"][param_name], no_mask_output["grads"][param_name], rtol=1e-4, atol=1e-4
                ),
                f"No-mask model should have same {param_name} updates as vanilla model",
            )
            self.assertTrue(
                torch.allclose(
                    vanilla_output["grads"][param_name],
                    default_output["grads"][param_name],
                    rtol=1e-4,
                    atol=1e-4,
                ),
                f"default forward run on the masked model should have same {param_name} updates as vanilla model",
            )

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
                    f"Masked dimensions of {param_name} should have zero updates",
                )

            for param_name in config["grads"][is_match]["non_zero"]:
                if param_name not in output["grads"]:
                    continue

                slice_fn = self.get_slice_function(param_name, slice_type_fn)
                grad = output["grads"][param_name]
                grad_slice = slice_fn(grad, retain_dim)

                self.assertTrue(
                    torch.all(torch.abs(grad_slice) > 0),
                    f"Non-masked dimensions of {param_name} should all be non-zero",
                )

            for param_name in config["grads"][is_match]["match_vanilla"]:
                if param_name not in output["grads"]:
                    continue

                slice_fn = self.get_slice_function(param_name, slice_type_fn)
                vanilla_grad = slice_fn(vanilla_output["grads"][param_name], retain_dim)
                masked_grad = slice_fn(output["grads"][param_name], retain_dim)

                self.assertTrue(
                    torch.allclose(vanilla_grad, masked_grad, rtol=1e-4, atol=1e-4),
                    f"Unmasked dimensions of {param_name} should match between vanilla and masked models",
                )

            for param_name in config["grads"][is_match]["mismatch_vanilla"]:
                if param_name not in output["grads"]:
                    continue

                slice_fn = self.get_slice_function(param_name, slice_type_fn)
                vanilla_grad = slice_fn(vanilla_output["grads"][param_name], retain_dim)
                masked_grad = slice_fn(output["grads"][param_name], retain_dim)

                self.assertFalse(
                    torch.allclose(vanilla_grad, masked_grad, rtol=1e-4, atol=1e-4),
                    f"Unmasked dimensions of {param_name} should match between vanilla and masked models",
                )


ACTIVATION_MASKING_CONFIG = {
    "forward_match": False,
    "grads": {
        False: {
            "zero": [
                "c_fc.weight",
                "c_fc.bias",
                "c_proj.weight",
                "mlp2.c_fc.weight",
                "mlp2.c_proj.weight",
            ],
            "non_zero": [],
            "match_vanilla": [],
            "mismatch_vanilla": [],
        },
        True: {
            "zero": [],
            "non_zero": [],
            "match_vanilla": [],
            "mismatch_vanilla": [
                "c_fc.weight",
                "c_fc.bias",
                "c_proj.weight",
                "mlp1.c_fc.weight",
                "mlp1.c_proj.weight",
                "mlp2.c_fc.weight",
                "mlp2.c_proj.weight",
            ],
        },
    },
}


class TestActivationMasking(BaseGPTNeoMLPTest):
    """Test class for the activation masking implementation"""

    def setUp(self):
        super().setUp()
        self.model_class = GPTNeoMLPActivationMasking
        self.split_masked_weights = False

        self.config.update(ACTIVATION_MASKING_CONFIG)

    def test_activation_masking_specific(self):
        """Test that masking is equivalent to zeroing weights"""
        retain_dim = self.intermediate_size // 4

        masked_model = self.create_model(GPTNeoMLPActivationMasking, retain_dim)  # 25% retain

        vanilla_manual_zero_fc_model = self.create_model(GPTNeoMLP, None)
        with torch.no_grad():
            vanilla_manual_zero_fc_model.c_fc.weight[:retain_dim] = 0
            vanilla_manual_zero_fc_model.c_fc.bias[:retain_dim] = 0

        forget_output = masked_model(self.input_data, sgtm_mode="forget")
        manual_zero_output = vanilla_manual_zero_fc_model(self.input_data)

        # Check that activation masking is equivalent to zeroing corresponding weights
        self.assertTrue(
            torch.allclose(manual_zero_output, forget_output),
            "Activation masking should be equivalent to zeroing the corresponding weights",
        )


class TestActivationMaskingSplit(TestActivationMasking):
    """Test class for the activation masking implementation"""

    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


class TestParameterMasking(BaseGPTNeoMLPTest):
    """Test class for the parameter masking implementation"""

    def setUp(self):
        super().setUp()
        self.model_class = GPTNeoMLPParameterMasking
        self.split_masked_weights = False

        self.config.update(
            {
                "forward_match": True,
                "grads": {
                    False: {
                        "zero": [
                            "c_fc.weight",
                            "c_fc.bias",
                            "c_proj.weight",
                            "mlp2.c_fc.weight",
                            "mlp2.c_proj.weight",
                        ],
                        "non_zero": [],
                        "match_vanilla": [
                            "mlp1.c_fc.weight",
                            "mlp1.c_proj.weight",
                        ],
                        "mismatch_vanilla": [],
                    },
                    True: {
                        "zero": [],
                        "non_zero": [],
                        "match_vanilla": [
                            "c_fc.weight",
                            "c_fc.bias",
                            "c_proj.weight",
                            "mlp1.c_fc.weight",
                            "mlp1.c_proj.weight",
                            "mlp2.c_fc.weight",
                            "mlp2.c_proj.weight",
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


class TestParameterMaskingNoProj(BaseGPTNeoMLPTest):
    """Test class for the parameter masking implementation"""

    def setUp(self):
        super().setUp()
        self.model_class = GPTNeoMLPParameterMaskingNoProj
        self.split_masked_weights = False

        self.config.update(
            {
                "forward_match": True,
                "grads": {
                    False: {
                        "zero": [
                            "c_fc.weight",
                            "c_fc.bias",
                            "mlp2.c_fc.weight",
                        ],
                        "non_zero": [],
                        "match_vanilla": [
                            "c_proj.weight",
                            "mlp2.c_proj.weight",
                            "mlp1.c_fc.weight",
                            "mlp1.c_proj.weight",
                        ],
                        "mismatch_vanilla": [],
                    },
                    True: {
                        "zero": [],
                        "non_zero": [],
                        "match_vanilla": [
                            "c_fc.weight",
                            "c_fc.bias",
                            "c_proj.weight",
                            "mlp1.c_fc.weight",
                            "mlp1.c_proj.weight",
                            "mlp2.c_fc.weight",
                            "mlp2.c_proj.weight",
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


class TestGradientRouting(BaseGPTNeoMLPTest):
    """Test class for the gradient routing implementation"""

    def setUp(self):
        super().setUp()
        self.model_class = GPTNeoMLPGradientRouting

        self.config.update(
            {
                "forward_match": True,
                "grads": {
                    False: {
                        "zero": [
                            "c_fc.weight",
                            "c_fc.bias",
                            "mlp2.c_fc.weight",
                        ],
                        "non_zero": [],
                        "match_vanilla": [
                            "c_proj.weight",
                            "mlp2.c_proj.weight",
                        ],
                        "mismatch_vanilla": [],
                    },
                    True: {
                        "zero": [],
                        "non_zero": [],
                        "match_vanilla": [
                            "c_fc.weight",
                            "c_fc.bias",
                            "c_proj.weight",
                            "mlp2.c_fc.weight",
                            "mlp2.c_proj.weight",
                        ],
                        "mismatch_vanilla": [
                            "mlp1.c_fc.weight",
                            "mlp1.c_proj.weight",
                        ],
                    },
                },
            }
        )


class TestGradientRoutingSplit(TestGradientRouting):
    """Test class for the parameter masking implementation"""

    def setUp(self):
        super().setUp()
        self.split_masked_weights = True


if __name__ == "__main__":
    unittest.main()
