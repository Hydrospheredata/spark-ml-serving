package io.hydrosphere.spark_ml_serving.common

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.Param

import scala.reflect.ClassTag

trait LocalModel[T <: Transformer] {
  final protected def loadParams(sparkTransformer: T, metadata: Metadata): Unit = {
    sparkTransformer.params.foreach{ p =>
      loadParam(sparkTransformer, metadata, p)
    }
  }

  final private def loadParam[P](sparkTransformer: T, metadata: Metadata, param: Param[P])(implicit ct: ClassTag[P]) = {
    val paramMap = metadata.paramMap ++ Map(
      "numFeatures" -> metadata.numFeatures,
      "numClasses" -> metadata.numClasses,
      "numTrees" -> metadata.numTrees
    )
    sparkTransformer.set(param.asInstanceOf[Param[Any]], ct.runtimeClass.cast(paramMap(param.name)))
  }

  def load(metadata: Metadata, data: LocalData): T
  implicit def getTransformer(transformer: T): LocalTransformer[T]
}
