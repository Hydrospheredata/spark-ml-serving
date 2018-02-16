package io.hydrosphere.spark_ml_serving.common

import org.apache.spark.ml.Transformer

trait TreeModelLoader[SparkConverter <: Transformer] extends ModelLoader[SparkConverter] {

  def build(metadata: Metadata, data: LocalData, treeMetadata: LocalData): SparkConverter

  def loadData(source: ModelSource, metadata: Metadata): LocalData = {
    ModelDataReader.parse(source, "data/")
  }

  def loadTreeMetadata(source: ModelSource): LocalData = {
    ModelDataReader.parse(source, "treesMetadata/")
  }

  def load(source: ModelSource): SparkConverter = {
    val trees       = loadTreeMetadata(source)
    val metadataRaw = source.readFile("metadata/part-00000")
    val metadata    = Metadata.fromJson(metadataRaw)
    val data        = loadData(source, metadata)
    build(metadata, data, trees)
  }
}
