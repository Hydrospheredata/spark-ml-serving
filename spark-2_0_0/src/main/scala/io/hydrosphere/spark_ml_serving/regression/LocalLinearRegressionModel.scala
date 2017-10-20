package io.hydrosphere.spark_ml_serving.regression

import io.hydrosphere.spark_ml_serving.common._
import DataUtils._
import io.hydrosphere.spark_ml_serving.common.DataUtils
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.LinearRegressionModel

class LocalLinearRegressionModel(override val sparkTransformer: LinearRegressionModel)
  extends LocalPredictionModel[LinearRegressionModel] {

}

object LocalLinearRegressionModel extends LocalModel[LinearRegressionModel] {
  override def load(metadata: Metadata, data: Map[String, Any]): LinearRegressionModel = {
    val intercept = data("intercept").asInstanceOf[java.lang.Double]
    val coeffitientsMap = data("coefficients").asInstanceOf[Map[String, Any]]
    val coeffitients = DataUtils.constructVector(coeffitientsMap)

    val ctor = classOf[LinearRegressionModel].getConstructor(classOf[String], classOf[Vector], classOf[Double])
    val inst = ctor.newInstance(metadata.uid, coeffitients, intercept)
    inst
      .set(inst.featuresCol, metadata.paramMap("featuresCol").asInstanceOf[String])
      .set(inst.predictionCol, metadata.paramMap("predictionCol").asInstanceOf[String])
      .set(inst.labelCol, metadata.paramMap("labelCol").asInstanceOf[String])
      .set(inst.elasticNetParam, metadata.paramMap("elasticNetParam").toString.toDouble)
      // NOTE: introduced in spark 2.1 for reducing iterations for big datasets, e.g unnecessary for us
      //.set(inst.aggregationDepth, metadata.paramMap("aggregationDepth").asInstanceOf[Int])
      .set(inst.maxIter, metadata.paramMap("maxIter").asInstanceOf[Number].intValue())
      .set(inst.regParam, metadata.paramMap("regParam").toString.toDouble)
      .set(inst.solver, metadata.paramMap("solver").asInstanceOf[String])
      .set(inst.tol, metadata.paramMap("tol").toString.toDouble)
      .set(inst.standardization, metadata.paramMap("standardization").asInstanceOf[Boolean])
      .set(inst.fitIntercept, metadata.paramMap("fitIntercept").asInstanceOf[Boolean])
  }

  override implicit def getTransformer(transformer: LinearRegressionModel): LocalTransformer[LinearRegressionModel] = new LocalLinearRegressionModel(transformer)
}
