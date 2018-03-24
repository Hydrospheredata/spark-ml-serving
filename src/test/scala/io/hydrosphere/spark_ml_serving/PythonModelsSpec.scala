package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.classification.LocalLogisticRegressionModel

class PythonModelsSpec extends GenericTestSpec {
  describe("LogisticRegression") {
    it("should load local version") {
      val localPipelineModel = LocalLogisticRegressionModel.load(getClass.getResource("/pyspark_models/py_log_reg").getPath)
    }
  }
}
