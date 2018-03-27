package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.linalg.{Vector, Vectors}

class LocalMultilayerPerceptronClassificationModel(
  override val sparkTransformer: MultilayerPerceptronClassificationModel
) extends LocalPredictionModel[MultilayerPerceptronClassificationModel] {}

object LocalMultilayerPerceptronClassificationModel
  extends SimpleModelLoader[MultilayerPerceptronClassificationModel]
  with TypedTransformerConverter[MultilayerPerceptronClassificationModel] {

  override def build(
    metadata: Metadata,
    data: LocalData
  ): MultilayerPerceptronClassificationModel = {
    val layers = data.column("layers").get.data.head.asInstanceOf[Seq[Int]].toArray
    val weightsParam = data.column("weights").get.data.head.asInstanceOf[Map[String, Any]]
    val weights = DataUtils.constructVector(weightsParam)
    val constructor = classOf[MultilayerPerceptronClassificationModel].getDeclaredConstructor(
      classOf[String],
      classOf[Array[Int]],
      classOf[Vector]
    )
    constructor.setAccessible(true)
    constructor
      .newInstance(
        metadata.uid,
        layers,
        weights
      )
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
  }

  override implicit def toLocal(
    sparkTransformer: MultilayerPerceptronClassificationModel
  ): LocalMultilayerPerceptronClassificationModel = {
    new LocalMultilayerPerceptronClassificationModel(sparkTransformer)
  }
}
