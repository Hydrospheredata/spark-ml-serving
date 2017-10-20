package io.hydrosphere.spark_ml_serving.common

import org.apache.spark.ml.Transformer

import scala.reflect.runtime.universe

object TransformerFactory {

  def apply(metadata: Metadata, data: Map[String, Any]): Transformer = {
    import io.hydrosphere.spark_ml_serving.ModelConversions._
    val runtimeMirror = universe.runtimeMirror(this.getClass.getClassLoader)
    val module = runtimeMirror.staticModule(metadata.`class`+ "$")
    val obj = runtimeMirror.reflectModule(module)
    val localModel = obj.instance
    localModel.load(metadata, data)
  }

}
