package ml.dmlc.xgboost4j.java;

import hex.tree.xgboost.BoosterParms;
import hex.tree.xgboost.XGBoostModel;
import water.*;
import water.nbhm.NonBlockingHashMap;
import water.util.Log;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

public class XGBoostUpdater extends Thread {

  private static long WORK_START_TIMEOUT_SECS = 5 * 60; // Each Booster iteration should before this timer expires
  private static long INACTIVE_CHECK_INTERVAL_SECS = 60;

  private static final NonBlockingHashMap<Key<XGBoostModel>, XGBoostUpdater> updaters = new NonBlockingHashMap<>();

  private final Key<XGBoostModel> _modelKey;
  private final DMatrix _trainMat;
  private final BoosterParms _boosterParms;
  private final Map<String, String> _rabitEnv;

  private SynchronousQueue<BoosterCallable<?>> _in;
  private SynchronousQueue<Object> _out;

  private Booster _booster;

  private XGBoostUpdater(Key<XGBoostModel> modelKey, DMatrix trainMat, BoosterParms boosterParms,
                         Map<String, String> rabitEnv) {
    super("XGBoostUpdater-" + modelKey);
    _modelKey = modelKey;
    _trainMat = trainMat;
    _boosterParms = boosterParms;
    _rabitEnv = rabitEnv;
    _in = new SynchronousQueue<>();
    _out = new SynchronousQueue<>();
  }

  @Override
  public void run() {
    try {
      Rabit.init(_rabitEnv);

      while (! interrupted()) {
        BoosterCallable<?> task = _in.take();
        Object result = task.call();
        _out.put(result);
      }
    } catch (InterruptedException e) {
      XGBoostUpdater self = updaters.get(_modelKey);
      if (self != null) {
        throw new IllegalStateException("Updater thread was interrupted while it was still registered, name=" + self.getName());
      }
    } catch (XGBoostError e) {
      throw new IllegalStateException("XGBoost training iteration failed", e);
    } finally {
      _in = null; // Will throw NPE if used wrong
      _out = null;
      updaters.remove(_modelKey);
      try {
        Rabit.shutdown();
      } catch (XGBoostError xgBoostError) {
        Log.warn("Rabit shutdown during update failed", xgBoostError);
      }
    }
  }

  private class UpdateBooster implements BoosterCallable<Booster> {
    private final int _tid;

    private UpdateBooster(int tid) { _tid = tid; }

    @Override
    public Booster call() throws XGBoostError {
      if ((_booster == null) && _tid == 0) {
        HashMap<String, DMatrix> watches = new HashMap<>();
        // Create empty Booster
        _booster = ml.dmlc.xgboost4j.java.XGBoost.train(_trainMat,
                _boosterParms.get(),
                0,
                watches,
                null,
                null);
        // Force Booster initialization; we can call any method that does "lazy init"
        byte[] boosterBytes = _booster.toByteArray();
        Log.info("Initial (0 tree) Booster created, size=" + boosterBytes.length);
      } else {
        // Do one iteration
        assert _booster != null;
        _booster.update(_trainMat, _tid);
      }
      return _booster;
    }
  }

  private class SerializeBooster implements BoosterCallable<byte[]> {
    @Override
    public byte[] call() throws XGBoostError {
      return _booster.toByteArray();
    }
  }

  Booster getBooster() {
    return _booster;
  }

  byte[] getBoosterBytes() throws InterruptedException {
    final SynchronousQueue<BoosterCallable<?>> inQ = _in;
    if (inQ == null)
      throw new IllegalStateException("Updater is inactive on node " + H2O.SELF);
    BoosterCallable serializeBooster = new SerializeBooster();
    inQ.offer(serializeBooster, WORK_START_TIMEOUT_SECS, TimeUnit.SECONDS);
    SynchronousQueue<Object> outQ;
    while ((outQ = _out) != null) {
      Object result = outQ.poll(INACTIVE_CHECK_INTERVAL_SECS, TimeUnit.SECONDS);
      if (result != null)
        return (byte[]) result;
    }
    throw new IllegalStateException("Cannot perform booster operation: updater is inactive on node " + H2O.SELF);
  }

  static void terminate(Key<XGBoostModel> modelKey) {
    XGBoostUpdater updater = updaters.remove(modelKey);
    if (updater == null)
      Log.debug("XGBoostUpdater for modelKey=" + modelKey + " was already clean-up on node " + H2O.SELF);
    else
      updater.interrupt();
  }

  private Booster doUpdate(int tid) throws InterruptedException {
    final SynchronousQueue<BoosterCallable<?>> inQ = _in;
    if (inQ == null)
      throw new IllegalStateException("Updater is inactive on node " + H2O.SELF);
    BoosterCallable updateBooster = new UpdateBooster(tid);
    inQ.offer(updateBooster, WORK_START_TIMEOUT_SECS, TimeUnit.SECONDS);
    SynchronousQueue<Object> outQ;
    while ((outQ = _out) != null) {
      Object result = outQ.poll(INACTIVE_CHECK_INTERVAL_SECS, TimeUnit.SECONDS);
      if (result != null)
        return (Booster) result;
    }
    throw new IllegalStateException("Cannot perform booster operation: updater is inactive on node " + H2O.SELF);
  }

  static Booster doUpdate(Key<XGBoostModel> modelKey, int tid) {
    XGBoostUpdater updater = getUpdater(modelKey);
    try {
      return updater.doUpdate(tid);
    } catch (InterruptedException e) {
      throw new IllegalStateException(e);
    }
  }

  static XGBoostUpdater getUpdater(Key<XGBoostModel> modelKey) {
    XGBoostUpdater updater = updaters.get(modelKey);
    if (updater == null) {
      throw new IllegalStateException("XGBoostUpdater for modelKey=" + modelKey + " was not found!");
    }
    return updater;
  }

  static XGBoostUpdater make(Key<XGBoostModel> modelKey, DMatrix trainMat, BoosterParms boosterParms,
                             Map<String, String> rabitEnv) {
    XGBoostUpdater updater = new XGBoostUpdater(modelKey, trainMat, boosterParms, rabitEnv);
    if (updaters.putIfAbsent(modelKey, updater) != null)
      throw new IllegalStateException("XGBoostUpdater for modelKey=" + modelKey + " already exists!");
    return updater;
  }

  private interface BoosterCallable<E> {
    E call() throws XGBoostError;
  }

}
