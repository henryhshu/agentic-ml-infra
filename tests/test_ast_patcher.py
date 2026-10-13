"""Unit tests and performance benchmarks for ASTCodeSynthesizer plugin."""
from __future__ import annotations

import time
import unittest
import torch
import torch.nn as nn

from agents.plugins.ast_patcher import (
    ASTCodeSynthesizer,
    compute_optimal_group_count,
)


class TestASTCodeSynthesizer(unittest.TestCase):
    """Test suite validating AST patching capabilities and latency constraints."""

    def setUp(self) -> None:
        self.synthesizer = ASTCodeSynthesizer(target_groups=32)

    def test_optimal_group_count(self) -> None:
        """Verify optimal divisor calculation for standard channel sizes."""
        self.assertEqual(compute_optimal_group_count(64, 32), 32)
        self.assertEqual(compute_optimal_group_count(32, 32), 32)
        self.assertEqual(compute_optimal_group_count(16, 32), 16)
        self.assertEqual(compute_optimal_group_count(24, 32), 24)
        self.assertEqual(compute_optimal_group_count(3, 32), 3)
        self.assertEqual(compute_optimal_group_count(7, 32), 7)
        self.assertEqual(compute_optimal_group_count(384, 32), 32)
        self.assertEqual(compute_optimal_group_count(0, 32), 1)

    def test_replace_batchnorm_with_groupnorm(self) -> None:
        """Test transformation of BatchNorm layers to GroupNorm in source code."""
        source_code = """
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        return self.fc(x.mean(dim=[2, 3]))
"""
        transformed, patch, count = self.synthesizer.replace_batchnorm_with_groupnorm(source_code)
        self.assertEqual(count, 2)
        self.assertIn("GroupNorm(32, 64", transformed)
        self.assertIn("GroupNorm(32, 128", transformed)
        self.assertNotIn("BatchNorm2d", transformed)
        self.assertNotIn("momentum", transformed)
        self.assertTrue(len(patch) > 0)
        self.assertTrue(self.synthesizer.validate_source(transformed))

        # Test execution of synthesized code
        local_scope = {}
        exec(transformed, globals(), local_scope)
        model_cls = local_scope["ConvNet"]
        model = model_cls()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_repair_linear_dimensions(self) -> None:
        """Test repairing mismatched linear projection dimensions in AST."""
        source_code = """
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        return self.fc(x)
"""
        transformed, patch, count = self.synthesizer.repair_linear_dimensions(
            source_code, old_in=1024, old_out=10, new_in=512, new_out=10
        )
        self.assertEqual(count, 1)
        self.assertIn("nn.Linear(512, 10)", transformed)
        self.assertNotIn("1024", transformed)
        self.assertTrue(len(patch) > 0)
        self.assertTrue(self.synthesizer.validate_source(transformed))

    def test_convert_module_batchnorm_to_groupnorm_in_memory(self) -> None:
        """Test live in-memory PyTorch module transformation with weight preservation."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10),
        )

        # Initialize with distinct weights
        with torch.no_grad():
            model[1].weight.fill_(2.5)
            model[1].bias.fill_(0.5)

        # Mutate module in shared RAM
        transformed_model = ASTCodeSynthesizer.convert_module_batchnorm_to_groupnorm(model)

        # Assert all BatchNorms are now GroupNorms
        bns = [m for m in transformed_model.modules() if isinstance(m, nn.BatchNorm2d)]
        gns = [m for m in transformed_model.modules() if isinstance(m, nn.GroupNorm)]
        self.assertEqual(len(bns), 0)
        self.assertEqual(len(gns), 2)

        # Assert weight transfer
        self.assertTrue(torch.allclose(gns[0].weight, torch.full_like(gns[0].weight, 2.5)))
        self.assertTrue(torch.allclose(gns[0].bias, torch.full_like(gns[0].bias, 0.5)))

        # Assert forward execution works
        x = torch.randn(2, 3, 16, 16)
        out = transformed_model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_performance_latency_sub_50ms(self) -> None:
        """Assert AST parsing, rewriting, compilation, and diffing execute in < 50ms."""
        complex_model_source = """
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([ResNetBlock(64, 64) for _ in range(8)])
        self.linear = nn.Linear(1024, 10)
"""
        iterations = 20
        times: list[float] = []

        for _ in range(iterations):
            t0 = time.perf_counter()
            _, patch, count = self.synthesizer.replace_batchnorm_with_groupnorm(complex_model_source)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)

        avg_latency_ms = sum(times) / len(times)
        max_latency_ms = max(times)
        print(f"\n[AST Performance Benchmark] Avg Latency: {avg_latency_ms:.3f} ms | Max Latency: {max_latency_ms:.3f} ms")

        # Must be well below the 50 ms budget
        self.assertLess(avg_latency_ms, 50.0, f"Average latency {avg_latency_ms:.2f}ms exceeds 50ms limit")
        self.assertLess(max_latency_ms, 50.0, f"Peak latency {max_latency_ms:.2f}ms exceeds 50ms limit")


if __name__ == "__main__":
    unittest.main()
