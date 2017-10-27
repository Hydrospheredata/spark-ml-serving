package io.hydrosphere.spark_ml_serving.common

import org.apache.spark.ml.{PipelineModel, Transformer}

import scala.collection.mutable

object ModelLoader {

  private val RandomForestClassifier = "org.apache.spark.ml.classification.RandomForestClassificationModel"
  private val GBTreeRegressor = "org.apache.spark.ml.regression.GBTRegressionModel"
  private val GBTreeClassifier = "org.apache.spark.ml.classification.GBTClassificationModel"
  private val RandomForestRegressor = "org.apache.spark.ml.regression.RandomForestRegressionModel"


  def get(source: ModelSource): PipelineModel = {
    val metadata = source.readFile("metadata/part-00000")
    val pipelineParameters = Metadata.fromJson(metadata)
    val stages: Array[Transformer] = getStages(pipelineParameters, source)
    val pipeline = TransformerFactory(
      pipelineParameters,
      LocalData(
        LocalDataColumn(
          "stages",
          stages.toList
        )
      )
    ).asInstanceOf[PipelineModel]
    pipeline
  }

  def getStages(pipelineParameters: Metadata, source: ModelSource): Array[Transformer] =
    pipelineParameters.paramMap("stageUids").asInstanceOf[List[String]].zipWithIndex.toArray.map {
      case (uid: String, index: Int) =>
        val currentStage = s"stages/${index}_$uid"
        val modelMetadata = source.readFile(s"$currentStage/metadata/part-00000")
        val stageParameters = Metadata.fromJson(modelMetadata)
        loadTransformer(stageParameters, source, currentStage)
    }

  def loadTransformer(stageParameters: Metadata, source: ModelSource, stagePath: String): Transformer = {
    stageParameters.`class` match {
      case RandomForestClassifier | RandomForestRegressor | GBTreeRegressor | GBTreeClassifier =>
        val data = ModelDataReader.parse(source, s"$stagePath/data")
        val treesMetadata = ModelDataReader.parse(source, s"$stagePath/treesMetadata")
        val newMetadata = Metadata(
          stageParameters.`class`,
          stageParameters.timestamp,
          stageParameters.sparkVersion,
          stageParameters.uid,
          stageParameters.paramMap,
          stageParameters.numFeatures,
          stageParameters.numClasses,
          stageParameters.numTrees,
          Some(treesMetadata)
        )
        TransformerFactory(newMetadata,data)
      case _ =>
        val data = ModelDataReader.parse(source, s"$stagePath/data/")
        TransformerFactory(stageParameters, data)
    }
  }

}
