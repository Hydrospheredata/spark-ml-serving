package io.hydrosphere.spark_ml_serving.common

import io.hydrosphere.spark_ml_serving.TransformerConverter
import org.apache.spark.ml.Transformer

trait LocalModel[T <: Transformer] extends LocalModelConverter {
  type SparkConverter = T
}

trait LocalModelConverter {
  type SparkConverter <: Transformer

  def load(metadata: Metadata, data: LocalData): SparkConverter
  implicit def getTransformer(transformer: SparkConverter): LocalTransformer[SparkConverter]

  implicit def toTransformer(transformer: Transformer)(implicit conv: TransformerConverter): LocalTransformer[SparkConverter] = {
    getTransformer(transformer.asInstanceOf[SparkConverter])
  }
}