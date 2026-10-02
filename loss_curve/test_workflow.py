"""Focused checks for architecture, preprocessing, gradients and loss bookkeeping."""

import unittest

import train_lanet as training
import numpy as np
import torch

from lanet_data import CSPBDataset, MODULATIONS
from lanet_model import CROP_LENGTHS, LANet, verify_reference_structure
from code_lag.model_wrapper import FeatureExtract


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.root = training.PROJECT.parent / "dataset" / "CSPB.ML" / "CSPB_ML_2022_Data"
        cls.dataset = CSPBDataset(cls.root, range(1, 21), limit=8)

    def test_structure_and_no_weight_loading(self):
        model = LANet()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        result = verify_reference_structure(model, training.REFERENCE_CHECKPOINT)
        self.assertEqual(result["trainable_parameters"], 3238312)
        self.assertFalse(result["weights_loaded"])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)

    def test_original_signal_and_bound_reading(self):
        raw = np.fromfile(self.root / "Batch_Dir_1" / "signal_1.tim", dtype=np.float32)
        np.testing.assert_array_equal(self.dataset.data[0, 2:].numpy(), raw[2::2] + 1j * raw[3::2])
        with (self.root / "CSPB_ML_2022_Signal_Truth_Labels.txt").open() as stream:
            original = next(stream).split(" ")
        symb = float(original[12]) / float(original[4]) / float(original[10])
        offset = float(original[6])
        expected = np.asarray([-symb + offset + 0.5, symb + offset + 0.5], dtype=np.float16)
        np.testing.assert_array_equal(self.dataset.data[0, :2].real.numpy(), expected.astype(np.float32))
        self.assertEqual(self.dataset.targets.tolist(), list(range(len(MODULATIONS))))

    def test_full_length_preprocess_matches_original(self):
        model = LANet().to(self.device).eval()
        reference = FeatureExtract({"noise_max": 0.3, "boi_thre": 14.0}).to(self.device).eval()
        data = self.dataset.data.to(self.device)
        with torch.no_grad():
            torch.testing.assert_close(model.preprocess(data), reference(data), rtol=0, atol=0)

    def test_all_lengths_and_joint_gradients(self):
        model = LANet().to(self.device).eval()
        data = self.dataset.data[:4].to(self.device)
        for length in CROP_LENGTHS:
            with torch.no_grad():
                result = model(data, test_len=length)
            self.assertEqual(result.shape, (4, 8))
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result.exp().sum(1), torch.ones(4, device=self.device))
        model.train()
        loss = torch.nn.functional.nll_loss(model(data), self.dataset.targets[:4].to(self.device))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for module in (model.backbone_iq, model.backbone_cc, model.lag):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_crop_candidates(self):
        extractor = LANet(noise_max=0).preprocess.to(self.device).train()
        data = self.dataset.data[:2].to(self.device)
        observed = set()
        torch.manual_seed(1234)
        for _ in range(80):
            _, cropped = extractor._stand_randn(data)
            observed.add(cropped.shape[-1])
        self.assertEqual(observed, set(CROP_LENGTHS))

    def test_weighted_loss_and_partial_window(self):
        accumulator = training.LossAccumulator()
        accumulator.add(2.0, 8, 3, 2048)
        accumulator.add(1.0, 4, 2, 32768)
        self.assertAlmostEqual(accumulator.loss, 20 / 12)
        self.assertAlmostEqual(accumulator.accuracy, 5 / 12)
        self.assertEqual(accumulator.steps, 2)
        self.assertEqual(sum(accumulator.length_batches.values()), 2)

    def test_write_boundary(self):
        with self.assertRaises(ValueError):
            training.inside_test(training.PROJECT / "outside.json")
        self.assertEqual(training.inside_test(training.ROOT / "losses"), training.ROOT / "losses")


if __name__ == "__main__":
    unittest.main(verbosity=2)
