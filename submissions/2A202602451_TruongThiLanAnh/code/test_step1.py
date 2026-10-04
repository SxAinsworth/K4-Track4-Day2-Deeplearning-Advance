"""Fast tests for the backbone-screening contract."""
import time
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from benchmark import bench
from step1_backbones import BACKBONES, _write_workbook


class TestStep1(unittest.TestCase):
    def test_backbone_coverage_and_unique_ids(self):
        ids=[row[0] for row in BACKBONES]; names=[row[1] for row in BACKBONES]
        self.assertGreaterEqual(len(names),5); self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(any("resnet" in name for name in names))
        self.assertTrue(any("convnext" in name or "resnext" in name for name in names))
        self.assertTrue(any(any(key in name for key in ("vit","deit","swin")) for name in names))
        self.assertTrue(any(any(key in name for key in ("efficientnet_b0","mobilenetv3")) for name in names))

    def test_benchmark_returns_percentiles(self):
        result=bench(lambda: time.sleep(0.001),warmup=1,iters=5)
        self.assertEqual(result["n"],5)
        self.assertLessEqual(result["p50"],result["p95"]); self.assertLessEqual(result["p95"],result["p99"])

    def test_backbones_workbook_is_created(self):
        frame=pd.DataFrame([{"exp_id":"B01","backbone":"resnet50","pretrained_tag":"resnet50.a1_in1k",
                             "params_m":25.6,"gmac":4.1,"val_macro_f1":0.9,"val_top1":0.95,
                             "train_seconds_epoch":10.0,"latency_p50_ms":5.0,"latency_p95_ms":6.0}])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"results.xlsx"; _write_workbook(frame,path)
            self.assertTrue(path.is_file())
            with pd.ExcelFile(path) as workbook:
                self.assertEqual(workbook.sheet_names,["Backbones"])


if __name__=="__main__": unittest.main()
