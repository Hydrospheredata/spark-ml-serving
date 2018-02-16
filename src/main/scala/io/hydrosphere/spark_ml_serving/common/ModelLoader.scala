package io.hydrosphere.spark_ml_serving.common

import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.spark.ml.Transformer

trait ModelLoader[T <: Transformer] {

  def load(source: ModelSource): T

  final def load(path: String): T = {
    val source = if (path.startsWith("hdfs://")) {
      val uri = new URI(path)
      val p   = uri.getPath
      ModelSource.hadoop(p, new Configuration())
    } else {
      ModelSource.local(path)
    }

    load(source)
  }
}
