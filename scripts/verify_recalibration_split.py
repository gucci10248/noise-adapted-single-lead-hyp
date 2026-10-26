"""Regression check: reconstruct 20/80 splits and independently verify Platt results.

Run after eval_external_recalibration.py; uses frozen predictions, no model training.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
R=Path(__file__).resolve().parents[1]
raw=pd.read_csv(R/'outputs/recalibration/recalibration_raw.csv')
for cohort,path in [('chapman','outputs/hyp_v1_aug/chapman_predictions.csv'),('georgia','outputs/georgia_ext/predictions.csv')]:
    df=pd.read_csv(R/path);y=df.y_hyp.to_numpy(int);z=df.logit.to_numpy(float)
    for rep,(evaluation,calibration) in enumerate(StratifiedShuffleSplit(n_splits=10,test_size=.2,random_state=42).split(z,y)):
        assert len(calibration)==int(np.ceil(.2*len(y)))
        assert not np.intersect1d(calibration,evaluation).size
        model=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000).fit(z[calibration,None],y[calibration])
        p=model.predict_proba(z[evaluation,None])[:,1]
        r=raw[(raw.cohort==cohort)&(raw.repeat==rep)&(raw.method=='platt')].iloc[0]
        assert r.n_calibration==len(calibration) and r.n_evaluation==len(evaluation)
        assert np.isclose(r.auroc,roc_auc_score(y[evaluation],p),rtol=0,atol=1e-10)
        assert np.isclose(r.platt_a,model.coef_[0,0],rtol=0,atol=1e-8)
        assert np.isclose(r.platt_b,model.intercept_[0],rtol=0,atol=1e-8)
    print(f'{cohort}: all 10 corrected splits and independent Platt fits verified')
