package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common.classification.LocalProbabilisticClassificationModel
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.linalg.{Matrix, Vector, Vectors}

class LocalNaiveBayes(override val sparkTransformer: NaiveBayesModel)
  extends LocalProbabilisticClassificationModel[NaiveBayesModel] {}

object LocalNaiveBayes
  extends SimpleModelLoader[NaiveBayesModel]
  with TypedTransformerConverter[NaiveBayesModel] {

  override def build(metadata: Metadata, data: LocalData): NaiveBayesModel = {
    val constructor = classOf[NaiveBayesModel].getDeclaredConstructor(
      classOf[String],
      classOf[Vector],
      classOf[Matrix]
    )
    constructor.setAccessible(true)
    val matrixMetadata = data.column("theta").get.data.head.asInstanceOf[Map[String, Any]]
    val matrix         = DataUtils.constructMatrix(matrixMetadata)
    val piParams = data.column("pi").get.data.head.asInstanceOf[Map[String, Any]]
    val piVec = DataUtils.constructVector(piParams)

    val nb = constructor
      .newInstance(metadata.uid, piVec, matrix)
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
      .setProbabilityCol(metadata.paramMap("probabilityCol").asInstanceOf[String])
      .setRawPredictionCol(metadata.paramMap("rawPredictionCol").asInstanceOf[String])

    nb.set(nb.smoothing, metadata.paramMap("smoothing").asInstanceOf[Number].doubleValue())
    nb.set(nb.modelType, metadata.paramMap("modelType").asInstanceOf[String])
    nb.set(nb.labelCol, metadata.paramMap("labelCol").asInstanceOf[String])

    nb
  }

  override implicit def toLocal(sparkTransformer: NaiveBayesModel): LocalNaiveBayes = {
    new LocalNaiveBayes(sparkTransformer)
  }
}
