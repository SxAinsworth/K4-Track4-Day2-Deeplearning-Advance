"""Correctness checks for the parts that fail silently: losses, CutMix, param groups, freezing, eval order."""
import unittest
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import losses, model, train
from losses import FocalLoss, mix_batch, mixed_loss


class Named(TensorDataset):
    def __getitem__(self, i):
        x, y = super().__getitem__(i); return x, int(y), f"img{i:03d}.jpg"


class TestStep0(unittest.TestCase):
    def test_focal_gamma_zero_equals_ce(self):
        torch.manual_seed(0); logits=torch.randn(16,9); labels=torch.randint(0,9,(16,))
        self.assertLess(abs(FocalLoss(0)(logits,labels).item()-F.cross_entropy(logits,labels).item()),1e-6)

    def test_cutmix_pixels_and_labels(self):
        torch.manual_seed(7); x=torch.arange(8*3*16*16,dtype=torch.float32).reshape(8,3,16,16); y=torch.arange(8)
        mixed,(a,b,lam)=mix_batch(x,y,1.0,"cutmix")
        self.assertTrue(torch.equal(a,y)); self.assertEqual(b.shape,y.shape); self.assertTrue(0<=lam<=1)
        self.assertTrue(lam==1 or not torch.equal(mixed,x))

    def test_cutmix_lambda_matches_pasted_area(self):
        # Constant images 0..7: the share of each image's own value is exactly lam.
        for seed in range(20):
            torch.manual_seed(seed); import numpy as np; np.random.seed(seed)
            x=torch.arange(8,dtype=torch.float32)[:,None,None,None].expand(8,3,32,32).clone()
            mixed,(a,b,lam)=mix_batch(x,torch.arange(8),1.0,"cutmix")
            i=int((a!=b).nonzero()[0])  # an image whose patch comes from a different image
            kept=(mixed[i,0]==x[i,0,0,0]).float().mean().item()
            self.assertAlmostEqual(kept,lam,places=6)

    def test_mixed_loss_definition(self):
        logits=torch.tensor([[3.,1.],[0.,2.]]); a=torch.tensor([0,1]); b=torch.tensor([1,0]); lam=.3
        ce=torch.nn.CrossEntropyLoss(); expected=lam*ce(logits,a)+(1-lam)*ce(logits,b)
        self.assertTrue(torch.allclose(mixed_loss(ce,logits,(a,b,lam)),expected))

    def test_class_weights_from_train_counts(self):
        counts=[100]*8+[900]; w=losses.class_weights(counts)
        self.assertAlmostEqual(w.sum().item(),9,places=5); self.assertAlmostEqual((w[0]/w[8]).item(),9,places=5)
        with self.assertRaises(ValueError): losses.class_weights([1]*8)


class TestModel(unittest.TestCase):
    def test_param_groups_three_groups(self):
        net=model.build_model("resnet18",pretrained=False); groups=model.param_groups(net,1e-4,1e-3,0.05)
        decay,no_decay,head=groups
        self.assertEqual((decay["lr"],decay["weight_decay"]),(1e-4,0.05))
        self.assertTrue(all(p.ndim>1 for p in decay["params"]))
        self.assertEqual((no_decay["lr"],no_decay["weight_decay"]),(1e-4,0.0))
        self.assertTrue(all(p.ndim<=1 for p in no_decay["params"]))
        self.assertEqual(head["lr"],1e-3)
        self.assertEqual({id(p) for p in head["params"]},{id(p) for p in net.get_classifier().parameters()})
        self.assertEqual(sum(len(g["params"]) for g in groups),len(list(net.parameters())))

    def test_frozen_backbone_only_head_trainable(self):
        net=model.build_model("resnet18",pretrained=False,init="frozen")
        groups=model.param_groups(net,1e-4,1e-3,0.05)
        self.assertEqual(len(groups),1); self.assertEqual(groups[0]["lr"],1e-3)
        net.train(); model.keep_frozen_bn_eval(net)
        self.assertFalse(net.bn1.training); self.assertTrue(net.get_classifier().training)


class TestTrain(unittest.TestCase):
    def test_evaluate_keeps_loader_order_and_eval_mode(self):
        torch.manual_seed(0); ds=Named(torch.randn(10,3,32,32),torch.arange(10)%9)
        net=model.build_model("resnet18",pretrained=False); net.train()
        names,y,logits,_=train.evaluate(net,DataLoader(ds,batch_size=3),torch.nn.CrossEntropyLoss(),torch.device("cpu"))
        self.assertEqual(names,[f"img{i:03d}.jpg" for i in range(10)])
        self.assertEqual(y.tolist(),(torch.arange(10)%9).tolist()); self.assertEqual(logits.shape,(10,9))
        self.assertFalse(net.training)

    def test_scheduler_warmup_then_cosine(self):
        p=torch.nn.Parameter(torch.zeros(1)); opt=torch.optim.AdamW([p],lr=1.0)
        s=train.build_scheduler(opt,train.Config(epochs=4,warmup_epochs=1),10); lrs=[]
        for _ in range(40): lrs.append(opt.param_groups[0]["lr"]); opt.step(); s.step()
        self.assertAlmostEqual(lrs[0],0.1); self.assertAlmostEqual(lrs[9],1.0)
        self.assertTrue(all(a>=b for a,b in zip(lrs[9:],lrs[10:]))); self.assertLess(lrs[-1],0.01)

    def test_parse_overrides(self):
        o=train.parse_overrides(["exp_id=B01","backbone=resnet50","seed=1","ema_decay=0.999","amp=false"])
        self.assertEqual(o,{"exp_id":"B01","backbone":"resnet50","seed":1,"ema_decay":0.999,"amp":False})
        with self.assertRaises(ValueError): train.parse_overrides(["nope=1"])

    def test_set_seed_reproducible(self):
        train.set_seed(3); a=torch.randn(4); train.set_seed(3); self.assertTrue(torch.equal(a,torch.randn(4)))


if __name__=="__main__": unittest.main()
