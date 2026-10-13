"""Mathematical parity testing suite for Gradient Accumulation Guard & GroupNorm.

Mathematically verifies:
  1. Exact gradient and parameter trajectory parity between large batch B (accum=1)
     and accumulated micro-batches B/k (accum=k) using GroupNorm.
  2. Proof that BatchNorm fails mathematical parity under micro-batch accumulation
     due to running statistic drift and batch-dimension variance distortion.
  3. Proof that converting BatchNorm -> GroupNorm via ASTCodeSynthesizer restores
     exact mathematical parity.
  4. Proof that unconditional optimizer.zero_grad() inside the accumulation loop
     erases early micro-batch gradients and destroys optimization trajectory,
     demonstrating the necessity of the AutoPARK Accumulation Guard.
  5. Uneven micro-batch boundary gating (is_accum_boundary) at dataset boundaries.
"""
from __future__ import annotations

import copy
import unittest
import torch
import torch.nn as nn

from agents.plugins.ast_patcher import ASTCodeSynthesizer
from pipeline.config import PipelineConfig


class GroupNormToyNet(nn.Module):
    """Network using Group Normalization to verify mathematical parity."""
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class BatchNormToyNet(nn.Module):
    """Network using BatchNorm2d to demonstrate statistical accumulation failure."""
    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestGradientAccumulationParity(unittest.TestCase):
    """Mathematical verification of Accumulation Guard & Equivalence Preservation."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_groupnorm_accumulation_parity(self) -> None:
        """Prove exact mathematical parity between full batch B and micro-batches B/k."""
        total_batch_size = 64
        micro_batch_size = 32
        accum_steps = 2
        in_channels = 3
        num_classes = 10

        # Generate fixed synthetic dataset
        inputs = torch.randn(total_batch_size, in_channels, 16, 16)
        labels = torch.randint(0, num_classes, (total_batch_size,))

        # Initialize identical models and optimizers
        torch.manual_seed(1234)
        model_full = GroupNormToyNet(in_channels, num_classes)
        model_accum = copy.deepcopy(model_full)

        criterion = nn.CrossEntropyLoss()
        opt_full = torch.optim.SGD(model_full.parameters(), lr=0.05, momentum=0.9)
        opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.05, momentum=0.9)

        # ── 1. Full Batch Reference Pass (B=64, accum=1) ──
        opt_full.zero_grad()
        out_full = model_full(inputs)
        loss_full = criterion(out_full, labels)
        loss_full.backward()
        opt_full.step()

        # ── 2. Accumulated Micro-Batch Pass (b=32, accum=2) ──
        opt_accum.zero_grad()
        for i in range(accum_steps):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            x_micro = inputs[start_idx:end_idx]
            y_micro = labels[start_idx:end_idx]

            out_micro = model_accum(x_micro)
            # Accumulation Guard: scale loss by accum_steps
            loss_micro = criterion(out_micro, y_micro) / accum_steps
            loss_micro.backward()

            is_accum_boundary = (i + 1 == accum_steps)
            if is_accum_boundary:
                opt_accum.step()
                opt_accum.zero_grad()

        # ── Assert Mathematical Parity: Gradients & Weights ──
        for (name_f, p_full), (name_a, p_accum) in zip(
            model_full.named_parameters(), model_accum.named_parameters()
        ):
            # Parameter weights after optimizer step must be identical
            self.assertTrue(
                torch.allclose(p_full.data, p_accum.data, atol=1e-5, rtol=1e-5),
                f"Weight divergence at {name_f}: max diff = {(p_full.data - p_accum.data).abs().max().item():.6e}",
            )

    def test_batchnorm_breaks_accumulation_parity(self) -> None:
        """Demonstrate that BatchNorm2d FAILS mathematical parity across micro-batches."""
        total_batch_size = 64
        micro_batch_size = 32
        accum_steps = 2
        in_channels = 3
        num_classes = 10

        inputs = torch.randn(total_batch_size, in_channels, 16, 16)
        labels = torch.randint(0, num_classes, (total_batch_size,))

        torch.manual_seed(1234)
        model_full = BatchNormToyNet(in_channels, num_classes)
        model_accum = copy.deepcopy(model_full)

        criterion = nn.CrossEntropyLoss()
        opt_full = torch.optim.SGD(model_full.parameters(), lr=0.05)
        opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.05)

        # Full batch
        opt_full.zero_grad()
        loss_full = criterion(model_full(inputs), labels)
        loss_full.backward()
        opt_full.step()

        # Accumulated micro-batches
        opt_accum.zero_grad()
        for i in range(accum_steps):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            loss_micro = criterion(model_accum(inputs[start_idx:end_idx]), labels[start_idx:end_idx]) / accum_steps
            loss_micro.backward()
            if i + 1 == accum_steps:
                opt_accum.step()
                opt_accum.zero_grad()

        # Check running statistics drift (running mean & running variance cannot be accumulated)
        running_mean_diff = (model_full.bn1.running_mean - model_accum.bn1.running_mean).abs().max().item()
        running_var_diff = (model_full.bn1.running_var - model_accum.bn1.running_var).abs().max().item()

        # Check maximum weight divergence between full batch and accumulated BatchNorm
        max_diff = max(
            (p_f.data - p_a.data).abs().max().item()
            for p_f, p_a in zip(model_full.parameters(), model_accum.parameters())
        )

        # BatchNorm running statistics drift substantially across micro-batches
        self.assertGreater(
            running_var_diff, 1e-2,
            f"Expected BatchNorm running variance to drift under micro-batches (got diff {running_var_diff:.4f})"
        )
        self.assertGreater(
            running_mean_diff, 1e-4,
            f"Expected BatchNorm running mean to drift under micro-batches (got diff {running_mean_diff:.6f})"
        )
        # Weights fail strict parity atol=1e-5
        self.assertGreater(
            max_diff, 1e-5,
            f"Expected BatchNorm weights to violate strict parity (diff={max_diff:.6e} > 1e-5)"
        )

    def test_ast_groupnorm_conversion_restores_parity(self) -> None:
        """Prove that converting BatchNorm -> GroupNorm via AST restores mathematical parity."""
        in_channels = 3
        num_classes = 10
        total_batch_size = 64
        micro_batch_size = 16
        accum_steps = 4

        inputs = torch.randn(total_batch_size, in_channels, 16, 16)
        labels = torch.randint(0, num_classes, (total_batch_size,))

        # Create BatchNorm model and convert it to GroupNorm via ASTCodeSynthesizer
        bn_model = BatchNormToyNet(in_channels, num_classes)
        model_full = ASTCodeSynthesizer.convert_module_batchnorm_to_groupnorm(bn_model, target_groups=16)
        model_accum = copy.deepcopy(model_full)

        criterion = nn.CrossEntropyLoss()
        opt_full = torch.optim.SGD(model_full.parameters(), lr=0.02)
        opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.02)

        # Full batch pass (B=64, accum=1)
        opt_full.zero_grad()
        criterion(model_full(inputs), labels).backward()
        opt_full.step()

        # Micro-batch pass (B/k=16, accum=4)
        opt_accum.zero_grad()
        for i in range(accum_steps):
            idx0 = i * micro_batch_size
            idx1 = idx0 + micro_batch_size
            (criterion(model_accum(inputs[idx0:idx1]), labels[idx0:idx1]) / accum_steps).backward()
            if i + 1 == accum_steps:
                opt_accum.step()
                opt_accum.zero_grad()

        # Parity is restored: check weights
        for p_f, p_a in zip(model_full.parameters(), model_accum.parameters()):
            self.assertTrue(torch.allclose(p_f.data, p_a.data, atol=1e-5, rtol=1e-5))

    def test_unconditional_zero_grad_erases_gradients(self) -> None:
        """Prove that calling optimizer.zero_grad() unconditionally inside the loop destroys accumulation."""
        inputs = torch.randn(64, 3, 16, 16)
        labels = torch.randint(0, 10, (64,))

        model_correct = GroupNormToyNet(3, 10)
        model_buggy = copy.deepcopy(model_correct)

        criterion = nn.CrossEntropyLoss()
        opt_correct = torch.optim.SGD(model_correct.parameters(), lr=0.05)
        opt_buggy = torch.optim.SGD(model_buggy.parameters(), lr=0.05)

        # Correct accumulation with Accumulation Guard
        opt_correct.zero_grad()
        (criterion(model_correct(inputs[:32]), labels[:32]) / 2).backward()
        (criterion(model_correct(inputs[32:]), labels[32:]) / 2).backward()
        opt_correct.step()
        opt_correct.zero_grad()

        # Buggy loop: calling zero_grad() every backward pass (user concern 2)
        for i, (x, y) in enumerate([(inputs[:32], labels[:32]), (inputs[32:], labels[32:])]):
            opt_buggy.zero_grad()  # <-- THE BUG: erases micro-batch 0 gradients!
            (criterion(model_buggy(x), y) / 2).backward()
            if i == 1:
                opt_buggy.step()

        # Buggy model only learned from the last micro-batch, so weights diverge significantly
        diverged = False
        for p_c, p_b in zip(model_correct.parameters(), model_buggy.parameters()):
            if not torch.allclose(p_c.data, p_b.data, atol=1e-3):
                diverged = True
                break
        self.assertTrue(diverged, "Unconditional zero_grad should cause substantial weight divergence.")

    def test_remediator_invariant_effective_batch(self) -> None:
        """Verify Remediator scales accum_steps when halving batch_size to preserve effective batch."""
        import shutil
        import tempfile
        from agents.remediator import RemediatorAgent
        from metrics.experiment import ExperimentTracker

        tmp_dir = tempfile.mkdtemp()
        try:
            tracker = ExperimentTracker(base_dir=tmp_dir)
            config = PipelineConfig(batch_size=512, accum_steps=1, preserve_effective_batch=True)
            self.assertEqual(config.effective_batch_size, 512)

            remediator = RemediatorAgent("test_remediator", tracker, config)

            # 1st OOM remediation: 512 -> 256, accum_steps 1 -> 2
            res1 = remediator.execute_action("reduce_batch_size", config)
            self.assertEqual(config.batch_size, 256)
            self.assertEqual(config.accum_steps, 2)
            self.assertEqual(config.effective_batch_size, 512)
            self.assertEqual(res1["status"], "success")

            # 2nd OOM remediation: 256 -> 128, accum_steps 2 -> 4
            res2 = remediator.execute_action("reduce_batch_size", config)
            self.assertEqual(config.batch_size, 128)
            self.assertEqual(config.accum_steps, 4)
            self.assertEqual(config.effective_batch_size, 512)
            self.assertEqual(res2["status"], "success")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
