package io.hydrosphere.spark_ml_serving.classification

import java.lang.Boolean

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.classification.LocalProbabilisticClassificationModel
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.{Matrix, SparseMatrix, Vector, Vectors}

class LocalLogisticRegressionModel(override val sparkTransformer: LogisticRegressionModel)
  extends LocalProbabilisticClassificationModel[LogisticRegressionModel] {}

object LocalLogisticRegressionModel
  extends SimpleModelLoader[LogisticRegressionModel]
  with TypedTransformerConverter[LogisticRegressionModel] {

  override def build(metadata: Metadata, data: LocalData): LogisticRegressionModel = {
    val constructor = classOf[LogisticRegressionModel].getDeclaredConstructor(
      classOf[String],
      classOf[Matrix],
      classOf[Vector],
      classOf[Int],
      java.lang.Boolean.TYPE
    )
    constructor.setAccessible(true)
    val coefficientMatrixParams =
      data.column("coefficientMatrix").get.data.head.asInstanceOf[Map[String, Any]]
    val coefficientMatrix = DataUtils.constructMatrix(coefficientMatrixParams)
    val interceptVectorParams =
      data.column("interceptVector").get.data.head.asInstanceOf[Map[String, Any]]
    val interceptVector = DataUtils.constructVector(interceptVectorParams)
    constructor
      .newInstance(
        metadata.uid,
        coefficientMatrix,
        interceptVector,
        data.column("numFeatures").get.data.head.asInstanceOf[java.lang.Integer],
        data.column("isMultinomial").get.data.head.asInstanceOf[java.lang.Boolean]
      )
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
      .setProbabilityCol(metadata.paramMap("probabilityCol").asInstanceOf[String])
      .setRawPredictionCol(metadata.paramMap("rawPredictionCol").asInstanceOf[String])
      .setThreshold(metadata.paramMap("threshold").asInstanceOf[Double])
  }

  override implicit def toLocal(
    transformer: LogisticRegressionModel
  ): LocalTransformer[LogisticRegressionModel] = new LocalLogisticRegressionModel(transformer)
}
