package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.common.DataUtils._
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vector

class LocalNormalizer(override val sparkTransformer: Normalizer) extends LocalTransformer[Normalizer] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val method = classOf[Normalizer].getMethod("createTransformFunc")
        val newData = column.data.mapToMlVectors.map { vector =>
          method.invoke(sparkTransformer).asInstanceOf[Vector => Vector](vector).toList
        }
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalNormalizer extends LocalModel[Normalizer] {
  override def load(metadata: Metadata, data: LocalData): Normalizer = {
    new Normalizer(metadata.uid)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
      .setP(metadata.paramMap("p").toString.toDouble)
  }

  override implicit def getTransformer(transformer: Normalizer): LocalTransformer[Normalizer] = new LocalNormalizer(transformer)
}
