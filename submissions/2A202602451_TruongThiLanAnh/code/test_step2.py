"""Design checks for controlled recipe ablations."""
import unittest
from train import Config
from step2_training import VARIANTS


class TestStep2(unittest.TestCase):
    def test_required_axes_and_values(self):
        axes={row[1] for row in VARIANTS}
        self.assertTrue({"A","B","C"} <= axes)
        for axis in ("A","B","C"):
            self.assertGreaterEqual(sum(row[1]==axis for row in VARIANTS),2)

    def test_each_ablation_changes_only_declared_fields(self):
        baseline=Config()
        allowed={
            "A":{"init"}, "B":{"aug","mix","mix_alpha"},
            "C":{"loss","label_smoothing","focal_gamma","class_weight_beta"}, "F":{"ema_decay"},
        }
        for exp_id,axis,_,overrides in VARIANTS:
            changed={key for key,value in overrides.items() if getattr(baseline,key)!=value}
            self.assertTrue(changed,exp_id); self.assertTrue(changed <= allowed[axis],(exp_id,changed))

    def test_t00_matches_required_baseline(self):
        cfg=Config()
        self.assertEqual((cfg.epochs,cfg.batch_size,cfg.lr_backbone,cfg.lr_head,cfg.weight_decay),
                         (12,64,1e-4,1e-3,0.05))
        self.assertEqual((cfg.init,cfg.aug,cfg.loss,cfg.seed),("finetune","basic","ce",0))


if __name__=="__main__": unittest.main()
