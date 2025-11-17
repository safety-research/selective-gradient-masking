import unittest
import torch
import torch.nn as nn

from sgtm.model.split_linear import SplitLinearOut


class TestSplitLinear(unittest.TestCase):
    """Test class for the SplitLinearOut implementation to ensure it's
    functionally equivalent to a standard nn.Linear layer."""

    def setUp(self):
        torch.manual_seed(42)

        # Define test dimensions
        self.batch_size = 8
        self.in_features = 16
        self.out_features = 32
        self.retain_dim = 8  # 25% of out_features

        # Test input
        self.input = torch.randn(self.batch_size, self.in_features)

    def test_parameter_count(self):
        """Test that SplitLinearOut has the same number of parameters as nn.Linear."""

        # Create models
        standard_linear = nn.Linear(self.in_features, self.out_features)
        split_linear = SplitLinearOut(self.in_features, self.out_features, self.retain_dim)

        # Count parameters
        standard_params = sum(p.numel() for p in standard_linear.parameters())
        split_params = sum(p.numel() for p in split_linear.parameters())

        self.assertEqual(
            standard_params, split_params, "SplitLinearOut should have the same number of parameters as nn.Linear"
        )

    def test_initialization_consistency(self):
        """Test that SplitLinearOut initializes parameters in a way that is
        consistent with nn.Linear when using the same seed."""

        # Set seed for both initializations
        torch.manual_seed(42)
        standard_linear = nn.Linear(self.in_features, self.out_features)

        # Get RNG state after standard_linear initialization
        standard_rng_state = torch.get_rng_state()

        torch.manual_seed(42)
        split_linear = SplitLinearOut(self.in_features, self.out_features, self.retain_dim)

        # Get RNG state after split_linear initialization
        split_rng_state = torch.get_rng_state()

        # Check that RNG states are identical after initialization
        self.assertTrue(
            torch.all(standard_rng_state == split_rng_state),
            "Random number generator state should be the same after initializing both models",
        )

        # Check that weights are identical before any updates
        self.assertTrue(
            torch.allclose(standard_linear.weight, split_linear.weight),
            "SplitLinearOut weight should initialize the same as nn.Linear",
        )

        self.assertTrue(
            torch.allclose(standard_linear.bias, split_linear.bias),
            "SplitLinearOut bias should initialize the same as nn.Linear",
        )

    def test_forward_pass_equivalence(self):
        """Test that SplitLinearOut forward pass is identical to nn.Linear
        when weights are identical."""

        # Create models with identical weights
        torch.manual_seed(42)
        standard_linear = nn.Linear(self.in_features, self.out_features)

        torch.manual_seed(42)
        split_linear = SplitLinearOut(self.in_features, self.out_features, self.retain_dim)

        # Run forward passes
        standard_output = standard_linear(self.input)
        split_output = split_linear(self.input)

        # Check outputs match
        self.assertTrue(
            torch.allclose(standard_output, split_output, rtol=1e-4, atol=1e-4),
            "SplitLinearOut forward pass should match nn.Linear",
        )

    def test_backward_pass_equivalence(self):
        """Test that SplitLinearOut backward pass updates parameters
        identically to nn.Linear."""

        torch.manual_seed(42)
        standard_linear = nn.Linear(self.in_features, self.out_features)

        torch.manual_seed(42)
        split_linear = SplitLinearOut(self.in_features, self.out_features, self.retain_dim)

        # Create identical target data
        target = torch.randn(self.batch_size, self.out_features)

        # Forward and backward passes for standard linear
        standard_output = standard_linear(self.input)
        standard_loss = nn.MSELoss()(standard_output, target)
        standard_loss.backward()

        # Forward and backward passes for split linear
        split_output = split_linear(self.input)
        split_loss = nn.MSELoss()(split_output, target)
        split_loss.backward()

        # Check that gradients match after reconstruction
        self.assertTrue(
            torch.allclose(
                standard_linear.weight.grad[: self.retain_dim], split_linear.weight_retain.grad, rtol=1e-4, atol=1e-4
            ),
            "Gradient for retain part of weight should match standard linear",
        )

        self.assertTrue(
            torch.allclose(
                standard_linear.weight.grad[self.retain_dim :], split_linear.weight_forget.grad, rtol=1e-4, atol=1e-4
            ),
            "Gradient for forget part of weight should match standard linear",
        )

        self.assertTrue(
            torch.allclose(
                standard_linear.bias.grad[: self.retain_dim], split_linear.bias_retain.grad, rtol=1e-4, atol=1e-4
            ),
            "Gradient for retain part of bias should match standard linear",
        )

        self.assertTrue(
            torch.allclose(
                standard_linear.bias.grad[self.retain_dim :], split_linear.bias_forget.grad, rtol=1e-4, atol=1e-4
            ),
            "Gradient for forget part of bias should match standard linear",
        )

    def test_update_and_consistency(self):
        """Test that after parameter updates, SplitLinearOut maintains
        consistent behavior with nn.Linear."""

        torch.manual_seed(42)
        standard_linear = nn.Linear(self.in_features, self.out_features)

        torch.manual_seed(42)
        split_linear = SplitLinearOut(self.in_features, self.out_features, self.retain_dim)

        # Create optimizers with same learning rate
        lr = 1.0
        standard_optimizer = torch.optim.SGD(standard_linear.parameters(), lr=lr)
        split_optimizer = torch.optim.SGD(split_linear.parameters(), lr=lr)

        # Create identical target data
        target = torch.randn(self.batch_size, self.out_features)

        # Training loop - perform several updates
        for i in range(5):
            # Standard linear update
            standard_optimizer.zero_grad()
            standard_output = standard_linear(self.input)
            standard_loss = nn.MSELoss()(standard_output, target)
            standard_loss.backward()
            standard_optimizer.step()

            # Split linear update
            split_optimizer.zero_grad()
            split_output = split_linear(self.input)
            split_loss = nn.MSELoss()(split_output, target)
            split_loss.backward()
            split_optimizer.step()

        # Check outputs still match after updates
        standard_output = standard_linear(self.input)
        split_output = split_linear(self.input)

        self.assertTrue(
            torch.allclose(standard_output, split_output, rtol=1e-4, atol=1e-4),
            "SplitLinearOut should still match nn.Linear after parameter updates",
        )

    def test_different_retain_dims(self):
        """Test that SplitLinearOut works properly with different retain_dim values."""

        for retain_dim in [0, 1, self.out_features // 2, self.out_features - 1, self.out_features]:
            # Create models
            torch.manual_seed(42)
            standard_linear = nn.Linear(self.in_features, self.out_features)

            torch.manual_seed(42)
            split_linear = SplitLinearOut(self.in_features, self.out_features, retain_dim)

            # Run forward passes
            standard_output = standard_linear(self.input)
            split_output = split_linear(self.input)

            # Check outputs match
            self.assertTrue(
                torch.allclose(standard_output, split_output, rtol=1e-4, atol=1e-4),
                f"SplitLinearOut with retain_dim={retain_dim} should match nn.Linear",
            )

    def test_parameters_method(self):
        """Test that parameters() and named_parameters() methods work correctly."""

        split_linear = SplitLinearOut(self.in_features, self.out_features, self.retain_dim)

        # Check parameters() returns the correct parameters
        parameters = list(split_linear.parameters())
        self.assertEqual(
            len(parameters),
            4,
            "Should have 4 parameters: weight_retain, weight_forget, bias_retain, bias_forget",
        )

        # Check parameter shapes
        self.assertEqual(
            parameters[0].shape,
            torch.Size([self.retain_dim, self.in_features]),
            "weight_retain shape should be (retain_dim, in_features)",
        )
        self.assertEqual(
            parameters[1].shape,
            torch.Size([self.out_features - self.retain_dim, self.in_features]),
            "weight_forget shape should be (out_features - retain_dim, in_features)",
        )
        self.assertEqual(
            parameters[2].shape, torch.Size([self.retain_dim]), "bias_retain shape should be (retain_dim,)"
        )
        self.assertEqual(
            parameters[3].shape,
            torch.Size([self.out_features - self.retain_dim]),
            "bias_forget shape should be (out_features - retain_dim,)",
        )

        # Check named_parameters() returns the correct parameter names and values
        named_params = dict(split_linear.named_parameters())
        self.assertEqual(len(named_params), 4, "Should have 4 named parameters")

        # Check keys in named parameters
        self.assertTrue("weight_retain" in named_params, "named_parameters should include 'weight_retain'")
        self.assertTrue("weight_forget" in named_params, "named_parameters should include 'weight_forget'")
        self.assertTrue("bias_retain" in named_params, "named_parameters should include 'bias_retain'")
        self.assertTrue("bias_forget" in named_params, "named_parameters should include 'bias_forget'")

        # Check that the weight and bias properties are not in the parameter list
        self.assertTrue("weight" not in named_params, "'weight' property should not be in named_parameters")
        self.assertTrue("bias" not in named_params, "'bias' property should not be in named_parameters")

        # Check total parameter count
        total_params = sum(p.numel() for p in split_linear.parameters())
        expected_params = self.in_features * self.out_features + self.out_features  # weights + biases
        self.assertEqual(total_params, expected_params, "Total parameter count should match expected")


if __name__ == "__main__":
    unittest.main()
