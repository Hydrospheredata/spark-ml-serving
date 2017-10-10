package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving._
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.linalg.{Matrix, Vector, Vectors}

class LocalNaiveBayes(override val sparkTransformer: NaiveBayesModel)
  extends LocalProbabilisticClassificationModel[NaiveBayesModel] {
}

object LocalNaiveBayes extends LocalModel[NaiveBayesModel] {
  override def load(metadata: Metadata, data: Map[String, Any]): NaiveBayesModel = {
    val constructor = classOf[NaiveBayesModel].getDeclaredConstructor(classOf[String], classOf[Vector], classOf[Matrix])
    constructor.setAccessible(true)

    val matrixMetadata = data("theta").asInstanceOf[Map[String, Any]]
    val matrix = DataUtils.constructMatrix(matrixMetadata)
    val pi = data("pi").
      asInstanceOf[Map[String, Any]].
      getOrElse("values", List()).
      asInstanceOf[List[Double]].toArray
    val piVec = Vectors.dense(pi)

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

  override implicit def getTransformer(transformer: NaiveBayesModel): LocalTransformer[NaiveBayesModel] = new LocalNaiveBayes(transformer)
}
