"""Unit tests for AutoPARK Friction-Minimizing Abstractions.

Verifies:
1. DataModule Abstraction (BaseDataModule, SyntheticVisionDataModule, CustomDataModule, factory)
2. Model Registry (@register_model, get_model, list_models, custom builder)
3. Dynamic Evaluation Stage (custom classes, non-CIFAR labels, dynamic confusion matrix)
4. High-Level AutoPARK SDK (AutoParkRunner, from_loaders)
5. End-to-end multi-dataset execution
"""
from __future__ import annotations

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autopark import (
    AutoParkRunner,
    BaseDataModule,
    CIFAR10DataModule,
    CustomDataModule,
    PipelineConfig,
    SyntheticVisionDataModule,
    create_data_module,
    get_model,
    list_models,
    register_model,
)
from pipeline.evaluation import EvaluationStage
from pipeline.runner import PipelineRunner
from metrics.experiment import ExperimentTracker


class TestAutoParkAbstractions(unittest.TestCase):
    """Test suite verifying modular data, model, and runner abstractions."""

    def test_datamodule_synthetic_contract(self) -> None:
        """Verify SyntheticVisionDataModule conforms to BaseDataModule contract."""
        dm = SyntheticVisionDataModule(
            num_classes=5,
            train_samples=50,
            val_samples=20,
            image_shape=(3, 32, 32),
            class_names=["cat0", "cat1", "cat2", "cat3", "cat4"],
        )
        dm.setup()
        self.assertEqual(dm.train_samples, 50)
        self.assertEqual(dm.val_samples, 20)
        self.assertEqual(dm.num_classes, 5)
        self.assertEqual(len(dm.class_names), 5)

        is_valid, issues = dm.validate_schema()
        self.assertTrue(is_valid, f"Validation failed with issues: {issues}")
        self.assertEqual(len(issues), 0)

        train_loader = dm.train_dataloader(batch_size=10)
        val_loader = dm.val_dataloader(batch_size=10)
        x_batch, y_batch = next(iter(train_loader))
        self.assertEqual(x_batch.shape, (10, 3, 32, 32))
        self.assertEqual(y_batch.shape, (10,))
        self.assertTrue((y_batch < 5).all())

    def test_datamodule_custom_dataset(self) -> None:
        """Verify CustomDataModule wrapping custom PyTorch TensorDatasets."""
        x_tr = torch.randn(30, 4)
        y_tr = torch.randint(0, 3, (30,))
        x_val = torch.randn(10, 4)
        y_val = torch.randint(0, 3, (10,))

        dm = CustomDataModule(
            train_dataset=TensorDataset(x_tr, y_tr),
            val_dataset=TensorDataset(x_val, y_val),
            num_classes=3,
            class_names=["alpha", "beta", "gamma"],
        )
        is_valid, issues = dm.validate_schema()
        self.assertTrue(is_valid)
        self.assertEqual(dm.train_samples, 30)
        self.assertEqual(dm.val_samples, 10)

    def test_create_data_module_factory(self) -> None:
        """Verify create_data_module factory resolution."""
        cfg_cifar = PipelineConfig(dataset="cifar10", subset=50)
        dm1 = create_data_module(cfg_cifar)
        self.assertIsInstance(dm1, CIFAR10DataModule)

        cfg_synth = PipelineConfig(dataset="synthetic", subset=50, num_classes=8)
        dm2 = create_data_module(cfg_synth)
        self.assertIsInstance(dm2, SyntheticVisionDataModule)
        self.assertEqual(dm2.num_classes, 8)

        custom_dm = SyntheticVisionDataModule(num_classes=4)
        cfg_custom = PipelineConfig(data_module=custom_dm)
        dm3 = create_data_module(cfg_custom)
        self.assertIs(dm3, custom_dm)

        with self.assertRaises(ValueError):
            create_data_module(PipelineConfig(dataset="nonexistent_dataset_format"))

    def test_model_registry_custom_registration(self) -> None:
        """Verify @register_model decorator and get_model retrieval."""
        @register_model("test_custom_classifier")
        def build_custom_classifier(num_classes: int = 10, **kwargs: any) -> nn.Module:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 32 * 3, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )

        self.assertIn("test_custom_classifier", list_models())
        model = get_model("test_custom_classifier", num_classes=7)
        self.assertIsInstance(model, nn.Module)

        # Test forward pass with dummy tensor
        dummy = torch.randn(2, 3, 32, 32)
        out = model(dummy)
        self.assertEqual(out.shape, (2, 7))

        # Unknown model error
        with self.assertRaises(KeyError):
            get_model("unknown_hypothetical_model_xyz")

    def test_evaluation_stage_dynamic_classes(self) -> None:
        """Verify EvaluationStage adapts to arbitrary class count and labels."""
        num_classes = 4
        class_names = ["north", "south", "east", "west"]
        dm = SyntheticVisionDataModule(
            num_classes=num_classes,
            train_samples=20,
            val_samples=20,
            class_names=class_names,
        )
        dm.setup()
        val_loader = dm.val_dataloader(batch_size=10)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, num_classes),
        )

        config = PipelineConfig(
            dataset="synthetic",
            num_classes=num_classes,
            data_module=dm,
            device="cpu",
        )
        import shutil
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        try:
            tracker = ExperimentTracker(base_dir=tmp_dir)
            stage = EvaluationStage(
                config=config,
                tracker=tracker,
                model=model,
                test_loader=val_loader,
            )

            self.assertEqual(stage.num_classes, 4)
            self.assertEqual(stage.class_names, class_names)

            result = stage.execute()
            self.assertIn("per_class_accuracy", result)
            for name in class_names:
                self.assertIn(name, result["per_class_accuracy"])
            self.assertEqual(len(result["confusion_matrix"]), 4)
            self.assertEqual(len(result["confusion_matrix"][0]), 4)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_autopark_sdk_from_loaders(self) -> None:
        """Verify AutoParkRunner.from_loaders executes an end-to-end custom pipeline in < 10 lines."""
        # MLE's existing data and model
        x_train = torch.randn(40, 3, 32, 32)
        y_train = torch.randint(0, 3, (40,))
        x_val = torch.randn(20, 3, 32, 32)
        y_val = torch.randint(0, 3, (20,))

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        runner = AutoParkRunner.from_loaders(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=3,
            class_names=["red", "green", "blue"],
            epochs=1,
            batch_size=16,
            device="cpu",
            enable_agentic=False,
        )

        res = runner.run()
        self.assertEqual(res["status"], "success")
        self.assertIn("training", res["stages_run"])
        self.assertIn("evaluation", res["stages_run"])

    def test_end_to_end_synthetic_pipeline_run(self) -> None:
        """Verify end-to-end PipelineRunner execution with synthetic vision dataset and MLP."""
        config = PipelineConfig(
            dataset="synthetic",
            model="mlp",
            subset=60,
            epochs=1,
            batch_size=20,
            device="cpu",
            enable_agentic=False,
            enable_tuning=False,
        )
        runner = PipelineRunner(config)
        summary = runner.run()
        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["stages_run"], ["data_validation", "data_loading", "training", "evaluation"])


if __name__ == "__main__":
    unittest.main()
