package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.DCT
import org.apache.spark.ml.linalg.Vector

class LocalDCT(override val sparkTransformer: DCT) extends LocalTransformer[DCT] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val method = classOf[DCT].getMethod("createTransformFunc")
        val newData = column.data.mapToMlVectors.map { r =>
          method.invoke(sparkTransformer).asInstanceOf[Vector => Vector](r).toList
        }
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalDCT extends SimpleModelLoader[DCT] with TypedTransformerConverter[DCT] {
  override def build(metadata: Metadata, data: LocalData): DCT = {
    new DCT(metadata.uid)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
      .setInverse(metadata.paramMap("inverse").asInstanceOf[Boolean])
  }

  override implicit def toLocal(transformer: DCT) = new LocalDCT(transformer)
}
