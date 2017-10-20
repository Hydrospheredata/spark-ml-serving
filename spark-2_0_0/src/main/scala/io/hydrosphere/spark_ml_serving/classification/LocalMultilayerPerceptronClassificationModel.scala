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
  override def load(metadata: Metadata, data: Map[String, Any]): MultilayerPerceptronClassificationModel = {
    val constructor = classOf[MultilayerPerceptronClassificationModel].getDeclaredConstructor(classOf[String], classOf[Array[Int]], classOf[Vector])
    constructor.setAccessible(true)
    constructor
      .newInstance(
        metadata.uid,
        data("layers").asInstanceOf[List[Int]].to[Array],
        Vectors.dense(data("weights").asInstanceOf[Map[String, Any]]("values").asInstanceOf[List[Double]].toArray)
      )
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
  }

  override implicit def getTransformer(transformer: MultilayerPerceptronClassificationModel): LocalTransformer[MultilayerPerceptronClassificationModel] = new LocalMultilayerPerceptronClassificationModel(transformer)
}
