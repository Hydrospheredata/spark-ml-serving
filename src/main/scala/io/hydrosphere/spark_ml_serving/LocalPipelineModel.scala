package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.{PipelineModel, Transformer}
import io.hydrosphere.spark_ml_serving.common.utils.PumpedClass

class LocalPipelineModel(override val sparkTransformer: PipelineModel)
  extends LocalTransformer[PipelineModel] {

  def transform(localData: LocalData): LocalData = {
    import CommonTransormerConversions._

    sparkTransformer.stages.foldLeft(localData) {
      case (data, transformer) =>
        transformer.transform(data)
    }
  }

}

object LocalPipelineModel
  extends ModelLoader[PipelineModel]
  with TypedTransformerConverter[PipelineModel] {

  import CommonLoaderConversions._

  def getStages(pipelineParameters: Metadata, source: ModelSource): Array[Transformer] = {
    pipelineParameters.paramMap("stageUids").asInstanceOf[List[String]].zipWithIndex.toArray.map {
      case (uid: String, index: Int) =>
        val currentStage    = s"stages/${index}_$uid"
        val modelMetadata   = source.readFile(s"$currentStage/metadata/part-00000")
        val stageParameters = Metadata.fromJson(modelMetadata)
        val companion       = PumpedClass.companionFromClassName(stageParameters.`class`)
        companion.load(s"${source.root}/$currentStage").asInstanceOf[Transformer]
    }
  }

  override def load(source: ModelSource): PipelineModel = {
    val metadata                   = source.readFile("metadata/part-00000")
    val pipelineParameters         = Metadata.fromJson(metadata)
    val stages: Array[Transformer] = getStages(pipelineParameters, source)
    val cstr = classOf[PipelineModel].getDeclaredConstructor(
      classOf[String],
      classOf[Array[Transformer]]
    )
    cstr.setAccessible(true)
    cstr
      .newInstance(
        pipelineParameters.uid,
        stages
      )
  }

  implicit def toLocal(sparkTransformer: PipelineModel) =
    new LocalPipelineModel(sparkTransformer)
}
