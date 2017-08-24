package io.hydrosphere.spark_ml_serving

import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.spark.ml.PipelineModel

object PipelineLoader {

  def load(path: String): PipelineModel = {
    if (path.startsWith("hdfs://")) {
      val uri = new URI(path)
      val p = uri.getPath
      val src = ModelSource.hadoop(p, new Configuration())
      ModelLoader.get(src)
    } else {
      ModelLoader.get(ModelSource.local(path))
    }
  }

  def load(source: ModelSource): PipelineModel = {
    ModelLoader.get(source)
  }

}
