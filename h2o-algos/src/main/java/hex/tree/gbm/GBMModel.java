package hex.tree.gbm;

import hex.Distributions;
import hex.tree.SharedTreeModel;
import water.Key;
import water.util.SB;

public class GBMModel extends SharedTreeModel<GBMModel,GBMModel.GBMParameters,GBMModel.GBMOutput> {

  public static class GBMParameters extends SharedTreeModel.SharedTreeParameters {
    /** Distribution functions.  Note: AUTO will select gaussian for
     *  continuous, and multinomial for categorical response
     *
     *  <p>TODO: Replace with drop-down that displays different distributions
     *  depending on cont/cat response
     */
    public enum Family {  AUTO, bernoulli, multinomial, gaussian, poisson, gamma }
    public Family _distribution = Family.AUTO;
    public float _learn_rate=0.1f; // Learning rate from 0.0 to 1.0
  }

  public static class GBMOutput extends SharedTreeModel.SharedTreeOutput {
    public GBMOutput( GBM b, double mse_train, double mse_valid ) { super(b,mse_train,mse_valid); }
  }

  public GBMModel(Key selfKey, GBMParameters parms, GBMOutput output ) { super(selfKey,parms,output); }

  /** Bulk scoring API for one row.  Chunks are all compatible with the model,
   *  and expect the last Chunks are for the final distribution and prediction.
   *  Default method is to just load the data into the tmp array, then call
   *  subclass scoring logic. */
  @Override protected double[] score0(double data[/*ncols*/], double preds[/*nclasses+1*/], double weight, double offset) {
    super.score0(data, preds, weight, offset);    // These are f_k(x) in Algorithm 10.4
    if (_parms._distribution == GBMParameters.Family.bernoulli) {
      double f = preds[1] + _output._init_f + offset; //Note: class 1 probability stored in preds[1] (since we have only one tree)
      preds[2] = Distributions.Bernoulli.linkInv(f);
      preds[1] = 1.0 - preds[2];
    } else if (_parms._distribution == GBMParameters.Family.multinomial) { // Kept the initial prediction for binomial
      if (_output.nclasses() == 2) { //1-tree optimization for binomial
        preds[1] += _output._init_f + offset; //offset is not yet allowed, but added here to be future-proof
        preds[2] = -preds[1];
      }
      hex.genmodel.GenModel.GBM_rescale(preds);
    } else { //Regression
      double f = preds[0] + _output._init_f + offset;
      if( _parms._distribution == GBMParameters.Family.gaussian) {
        preds[0] = Distributions.Gaussian.linkInv(f);
      } else if( _parms._distribution == GBMParameters.Family.poisson) {
        preds[0] = Distributions.Poisson.linkInv(f);
      } else if( _parms._distribution == GBMParameters.Family.gamma) {
        preds[0] = Distributions.Gamma.linkInv(f);
      }
    }
    return preds;
  }

  // Note: POJO scoring code doesn't support per-row offsets (the scoring API would need to be changed to pass in offsets)
  @Override protected void toJavaUnifyPreds(SB body, SB file) {
    // Preds are filled in from the trees, but need to be adjusted according to
    // the loss function.
    if( _parms._distribution == GBMParameters.Family.bernoulli ) {
      body.ip("double fx = preds[1] + ").p(_output._init_f).p(";").nl();
      body.ip("preds[2] = 1.0/(1.0+Math.exp(-fx));").nl();
      body.ip("preds[1] = 1.0-preds[2];").nl();
      if (_parms._balance_classes)
        body.ip("hex.genmodel.GenModel.correctProbabilities(preds, PRIOR_CLASS_DISTRIB, MODEL_CLASS_DISTRIB);").nl();
      body.ip("preds[0] = hex.genmodel.GenModel.getPrediction(preds, data, " + defaultThreshold() + ");").nl();
      return;
    }
    if( _output.nclasses() == 1 ) { // Regression
      if( _parms._distribution == GBMParameters.Family.gaussian) {
        // Prediction starts from the mean response, and adds predicted residuals
        body.ip("preds[0] += ").p(_output._init_f).p(";");
      } else if( _parms._distribution == GBMParameters.Family.poisson) {
        body.ip("preds[0] += ").p(_output._init_f).p(";");
        body.ip("preds[0] = Math.max(1e-19,Math.min(1e19,Math.exp(preds[0])));");
      } else if( _parms._distribution == GBMParameters.Family.gamma) {
        body.ip("preds[0] += ").p(_output._init_f).p(";");
        body.ip("preds[0] = -1./preds[0];");
      }
      return;
    }
    if( _output.nclasses()==2 ) { // Kept the initial prediction for binomial
      body.ip("preds[1] += ").p(_output._init_f).p(";").nl();
      body.ip("preds[2] = - preds[1];").nl();
    }
    body.ip("hex.genmodel.GenModel.GBM_rescale(preds);").nl();
    if (_parms._balance_classes)
      body.ip("hex.genmodel.GenModel.correctProbabilities(preds, PRIOR_CLASS_DISTRIB, MODEL_CLASS_DISTRIB);").nl();
    body.ip("preds[0] = hex.genmodel.GenModel.getPrediction(preds, data, " + defaultThreshold() + ");").nl();
  }
}
