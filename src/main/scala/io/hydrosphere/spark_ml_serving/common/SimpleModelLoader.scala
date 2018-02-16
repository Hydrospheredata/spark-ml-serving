package io.hydrosphere.spark_ml_serving.common

import org.apache.spark.ml.Transformer

trait SimpleModelLoader[SparkConverter <: Transformer] extends ModelLoader[SparkConverter] {

  def build(metadata: Metadata, data: LocalData): SparkConverter

  def getData(source: ModelSource, metadata: Metadata): LocalData = {
    ModelDataReader.parse(source, "data/")
  }

  def load(source: ModelSource): SparkConverter = {
    val metadataRaw = source.readFile("metadata/part-00000")
    val metadata    = Metadata.fromJson(metadataRaw)
    val data        = getData(source, metadata)
    build(metadata, data)
  }
}
