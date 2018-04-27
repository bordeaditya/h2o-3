package ml.dmlc.xgboost4j.java;

import hex.tree.xgboost.BoosterParms;
import hex.tree.xgboost.XGBoostModel;
import water.*;
import water.util.Log;

import java.io.ByteArrayInputStream;
import java.io.IOException;

public class XGBoostUpdateTask extends AbstractXGBoostTask<XGBoostUpdateTask> {

    private final int _tid;

    public XGBoostUpdateTask(XGBoostSetupTask setupTask, int tid) {
        super(setupTask);
        _tid = tid;
    }

    @Override
    protected void execute() {
        Booster booster = XGBoostUpdater.doUpdate(_modelKey, _tid);
        if (booster == null)
            throw new IllegalStateException("Boosting iteration didn't produce a valid Booster.");
    }

    // This is called from driver
    public Booster getBooster(BoosterParms boosterParms) {
        final H2ONode boosterNode = getBoosterNode();
        final Booster booster;
        if (H2O.SELF.equals(boosterNode)) {
            booster = XGBoostUpdater.getUpdater(_modelKey).getBooster();
        } else {
            Log.debug("Booster will be retrieved from a remote node, node=" + boosterNode);
            FetchBoosterTask t = new FetchBoosterTask(_modelKey);
            booster = new RPC<>(boosterNode, t).call().get().rawBooster();
            try {
                booster.setParams(boosterParms.get());
            } catch (XGBoostError e) {
                throw new IllegalStateException("Failed to re-initialize Booster", e);
            }
        }
        return booster;
    }

    private static class FetchBoosterTask extends DTask<FetchBoosterTask> {
        private final Key<XGBoostModel> _modelKey;

        // OUT
        private byte[] _boosterBytes;

        public FetchBoosterTask(Key<XGBoostModel> modelKey) {
            _modelKey = modelKey;
        }

        private Booster rawBooster() {
            try {
                Booster booster = Booster.loadModel(new ByteArrayInputStream(_boosterBytes));
                Log.debug("Booster created from bytes, raw size = " + _boosterBytes.length);
                return booster;
            } catch (XGBoostError | IOException xgBoostError) {
                throw new IllegalStateException("Failed to load the booster.", xgBoostError);
            }
        }

        @Override
        public void compute2() {
            XGBoostUpdater updater = XGBoostUpdater.getUpdater(_modelKey);
            try {
                _boosterBytes = updater.getBoosterBytes();
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
            tryComplete();
        }
    }

}
