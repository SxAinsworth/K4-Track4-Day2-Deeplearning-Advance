"""Fast unit tests for inference aggregation, calibration and BN fusion."""
import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import inference


class TestStep3(unittest.TestCase):
    def test_views(self):
        x=torch.arange(2*3*8*10).reshape(2,3,8,10)
        self.assertTrue(torch.equal(inference.view_hflip(x),x.flip(-1)))
        crops=inference.views_multicrop(x,6)
        self.assertEqual(len(crops),5)
        self.assertTrue(all(tuple(crop.shape)==(2,3,6,6) for crop in crops))

    def test_probability_and_logit_aggregation(self):
        first=np.array([[4.0,1.0],[0.0,3.0]])
        second=np.array([[3.0,2.0],[1.0,2.0]])
        for space in ("prob","logit"):
            result=inference.aggregate_views([first,second],space)
            np.testing.assert_allclose(result.sum(1),1.0,atol=1e-7)
            self.assertEqual(result.shape,(2,2))

    def test_temperature_is_positive_and_does_not_increase_nll(self):
        logits=np.array([[8.,0.],[5.,0.],[0.,7.],[0.,6.]],dtype=np.float64)
        labels=np.array([0,1,1,0])
        before=F.cross_entropy(torch.tensor(logits),torch.tensor(labels)).item()
        temperature=inference.fit_temperature(logits,labels)
        after=F.nll_loss(torch.tensor(np.log(inference.apply_temperature(logits,temperature))),torch.tensor(labels)).item()
        self.assertGreater(temperature,0)
        self.assertLessEqual(after,before+1e-7)

    def test_ensemble_is_normalized(self):
        a=np.array([[.8,.2],[.1,.9]]); b=np.array([[.6,.4],[.3,.7]])
        result=inference.ensemble_probs([a,b])
        np.testing.assert_allclose(result.sum(1),1.0)

    def test_conv_bn_fusion_preserves_output(self):
        torch.manual_seed(0)
        model=nn.Sequential(nn.Conv2d(3,5,3,padding=1,bias=False),nn.BatchNorm2d(5),nn.ReLU()).eval()
        x=torch.randn(4,3,16,16)
        fused=inference.fuse_conv_bn(model)
        self.assertFalse(any(isinstance(module,nn.BatchNorm2d) for module in fused.modules()))
        torch.testing.assert_close(model(x),fused(x),rtol=1e-5,atol=1e-5)


if __name__=="__main__": unittest.main()
