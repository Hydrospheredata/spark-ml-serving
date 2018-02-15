package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.{PipelineModel, Transformer}

import scala.collection.JavaConversions._

class LocalPipelineModel(override val sparkTransformer: PipelineModel)(implicit conv: TransformerConverter) extends LocalTransformer[PipelineModel] {
  def transform(localData: LocalData): LocalData = {
    import conv._

    sparkTransformer.stages.foldLeft(localData) {
      case (data, transformer) =>
        transformer.transform(data)
    }
  }
}

object LocalPipelineModel {
  implicit def getTransformer(transformer: PipelineModel)(implicit conv: TransformerConverter): LocalPipelineModel = new LocalPipelineModel(transformer)
}

class LocalPipelineModelLoader(implicit conv: TransformerConverter) extends LocalModel[PipelineModel] {
  override def load(metadata: Metadata, data: LocalData): PipelineModel = {
    val constructor = classOf[PipelineModel].getDeclaredConstructor(classOf[String], classOf[java.util.List[Transformer]])
    constructor.setAccessible(true)
    val stages: java.util.List[Transformer] = data.column("stages").get.data.asInstanceOf[List[Transformer]]
    constructor.newInstance(metadata.uid, stages)
  }

  implicit def getTransformer(transformer: PipelineModel): LocalPipelineModel = new LocalPipelineModel(transformer)
}
