package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.Binarizer

class LocalBinarizer(override val sparkTransformer: Binarizer) extends LocalTransformer[Binarizer] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val treshhold = sparkTransformer.getThreshold
        val newData = column.data.map(r => {
          if (r.asInstanceOf[Number].doubleValue() > treshhold) 1.0 else 0.0
        })
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalBinarizer
  extends SimpleModelLoader[Binarizer]
  with TypedTransformerConverter[Binarizer] {
  override def build(metadata: Metadata, data: LocalData): Binarizer = {
    new Binarizer(metadata.uid)
      .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
      .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
      .setThreshold(metadata.paramMap("threshold").toString.toDouble)
  }

  override implicit def toLocal(transformer: Binarizer) =
    new LocalBinarizer(transformer)
}
