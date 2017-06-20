package io.hydrosphere.spark_ml_serving

import org.apache.spark.ml.PipelineModel

object PipelineLoader {
  def load(path: String): PipelineModel = ModelLoader.get(path)
}
