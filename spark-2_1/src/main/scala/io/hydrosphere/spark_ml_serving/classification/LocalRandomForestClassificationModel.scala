package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.classification.LocalProbabilisticClassificationModel
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, RandomForestClassificationModel}

class LocalRandomForestClassificationModel(override val sparkTransformer: RandomForestClassificationModel)
  extends LocalProbabilisticClassificationModel[RandomForestClassificationModel] {
}

object LocalRandomForestClassificationModel extends LocalModel[RandomForestClassificationModel] {
  override def load(metadata: Metadata, data: LocalData): RandomForestClassificationModel = {
    val dataRows = data.toMapList
    val treesMetadata = metadata.treesMetadata.get.toMapList
    val trees = treesMetadata map { treeRow =>
      val meta = Metadata.fromJson(treeRow("metadata").toString)
        .copy(
          numFeatures = metadata.numFeatures,
          numClasses = metadata.numClasses
        )
      val treeNodesData = dataRows.filter(_("treeID") == treeRow("treeID")).map(_("nodeData")).asInstanceOf[List[Map[String, Any]]]
      LocalDecisionTreeClassificationModel.createTree(
        meta,
        LocalData.fromMapList(treeNodesData)
      )
    }
    val ctor = classOf[RandomForestClassificationModel].getDeclaredConstructor(classOf[String], classOf[Array[DecisionTreeClassificationModel]], classOf[Int], classOf[Int])
    ctor.setAccessible(true)
    ctor
      .newInstance(
        metadata.uid,
        trees.to[Array],
        metadata.numFeatures.get.asInstanceOf[java.lang.Integer],
        metadata.numClasses.get.asInstanceOf[java.lang.Integer]
      )
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
      .setProbabilityCol(metadata.paramMap("probabilityCol").asInstanceOf[String])
      .setRawPredictionCol(metadata.paramMap("rawPredictionCol").asInstanceOf[String])
  }

  override implicit def getTransformer(transformer: RandomForestClassificationModel): LocalTransformer[RandomForestClassificationModel] = new LocalRandomForestClassificationModel(transformer)
}