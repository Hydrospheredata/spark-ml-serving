package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.linalg.{Vector, Vectors}

class LocalMultilayerPerceptronClassificationModel(override val sparkTransformer: MultilayerPerceptronClassificationModel) extends LocalTransformer[MultilayerPerceptronClassificationModel] {

  override def transform(localData: LocalData): LocalData = {
    import DataUtils._
    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        val method = classOf[MultilayerPerceptronClassificationModel].getMethod("predict", classOf[Vector])
        method.setAccessible(true)
        val newColumn = LocalDataColumn(
          sparkTransformer.getPredictionCol,
          column.data.mapToMlVectors.map {
            method.invoke(sparkTransformer, _).asInstanceOf[Double]
          })
        localData.withColumn(newColumn)
      case None => localData
    }
  }
}

object LocalMultilayerPerceptronClassificationModel extends LocalModel[MultilayerPerceptronClassificationModel] {
  override def load(metadata: Metadata, data: LocalData): MultilayerPerceptronClassificationModel = {
    val layers = data.column("layers").get.data.head.asInstanceOf[List[Int]].toArray
    val weights = data.column("weights").get.data.head.asInstanceOf[Map[String, Any]]
    val constructor = classOf[MultilayerPerceptronClassificationModel].getDeclaredConstructor(classOf[String], classOf[Array[Int]], classOf[Vector])
    constructor.setAccessible(true)
    constructor
      .newInstance(
        metadata.uid,
        layers,
        Vectors.dense(weights("values").asInstanceOf[List[Double]].toArray)
      )
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
  }

  override implicit def getTransformer(transformer: MultilayerPerceptronClassificationModel): LocalTransformer[MultilayerPerceptronClassificationModel] = new LocalMultilayerPerceptronClassificationModel(transformer)
}
