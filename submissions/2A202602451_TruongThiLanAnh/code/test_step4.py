"""Fast tests for final prediction aggregation and calibration invariants."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

import inference
from eval import save_predictions
from step4_final import _load_ensemble_predictions


class TestStep4(unittest.TestCase):
    def test_temperature_scaling_preserves_predicted_class(self):
        logits=np.array([[4.,1.,0.],[0.,2.,1.]],dtype=np.float64)
        before=inference.apply_temperature(logits,1.0)
        after=inference.apply_temperature(logits,2.5)
        np.testing.assert_array_equal(before.argmax(1),after.argmax(1))

    def test_three_seed_predictions_are_averaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"predictions").mkdir()
            names=["a.jpg","b.jpg"]; labels=np.array([0,1])
            probabilities=[]
            for seed,value in enumerate((.70,.75,.80)):
                probs=np.full((2,9),(1-value)/8); probs[0,0]=value; probs[1,1]=value
                probabilities.append(probs)
                save_predictions(root/"predictions"/f"F01_seed{seed}_test.csv",names,labels,probs)
            got_names,got_labels,got_probs=_load_ensemble_predictions(root,"F01")
            np.testing.assert_array_equal(got_names,names)
            np.testing.assert_array_equal(got_labels,labels)
            np.testing.assert_allclose(got_probs,np.mean(probabilities,axis=0))


if __name__=="__main__": unittest.main()
