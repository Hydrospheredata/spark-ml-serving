package io.hydrosphere.spark_ml_serving.classification

import java.lang.Boolean

import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.classification.LocalProbabilisticClassificationModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.{Matrix, SparseMatrix, Vector, Vectors}

class LocalLogisticRegressionModel(override val sparkTransformer: LogisticRegressionModel)
  extends LocalProbabilisticClassificationModel[LogisticRegressionModel] {

}

object LocalLogisticRegressionModel extends LocalModel[LogisticRegressionModel] {
  override def load(metadata: Metadata, data: LocalData): LogisticRegressionModel = {
    val constructor = classOf[LogisticRegressionModel].getDeclaredConstructor(
      classOf[String],
      classOf[Vector],
      classOf[Double]
    )
    constructor.setAccessible(true)
    val coefficientsParams = data.column("coefficients").get.data.head.asInstanceOf[Map[String, Any]]
    val coefficients = DataUtils.constructVector(coefficientsParams)
    constructor
      .newInstance(metadata.uid, coefficients, data.column("intercept").get.data.head.asInstanceOf[java.lang.Double])
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
      .setProbabilityCol(metadata.paramMap("probabilityCol").asInstanceOf[String])
      .setThreshold(metadata.paramMap("threshold").asInstanceOf[Double])
  }

  override implicit def getTransformer(transformer: LogisticRegressionModel): LocalTransformer[LogisticRegressionModel] = new LocalLogisticRegressionModel(transformer)
}
