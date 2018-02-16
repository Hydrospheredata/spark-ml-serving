package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.classification.LocalProbabilisticClassificationModel
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.Vector

class LocalLogisticRegressionModel(override val sparkTransformer: LogisticRegressionModel)
  extends LocalProbabilisticClassificationModel[LogisticRegressionModel] {}

object LocalLogisticRegressionModel
  extends SimpleModelLoader[LogisticRegressionModel]
  with TypedTransformerConverter[LogisticRegressionModel] {

  override def build(metadata: Metadata, data: LocalData): LogisticRegressionModel = {
    val constructor = classOf[LogisticRegressionModel].getDeclaredConstructor(
      classOf[String],
      classOf[Vector],
      classOf[Double]
    )
    constructor.setAccessible(true)
    val coefficientsParams =
      data.column("coefficients").get.data.head.asInstanceOf[Map[String, Any]]
    val coefficients = DataUtils.constructVector(coefficientsParams)
    constructor
      .newInstance(
        metadata.uid,
        coefficients,
        data.column("intercept").get.data.head.asInstanceOf[java.lang.Double]
      )
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
      .setProbabilityCol(metadata.paramMap("probabilityCol").asInstanceOf[String])
      .setThreshold(metadata.paramMap("threshold").asInstanceOf[Double])
  }

  override implicit def toLocal(
    sparkTransformer: LogisticRegressionModel
  ): LocalLogisticRegressionModel = {
    new LocalLogisticRegressionModel(sparkTransformer)
  }
}
