package io.hydrosphere.spark_ml_serving.classification

import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.regression.LocalDecisionTreeRegressionModel
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.regression.DecisionTreeRegressionModel

class LocalGBTClassificationModel(override val sparkTransformer: GBTClassificationModel)
  extends LocalPredictionModel[GBTClassificationModel] { }

object LocalGBTClassificationModel extends LocalModel[GBTClassificationModel] {
  override def load(metadata: Metadata, data: LocalData): GBTClassificationModel = {
    val dataRows = data.toMapList
    val treesMetadata = metadata.treesMetadata.get.toMapList
    val trees = treesMetadata map { treeRow =>
      val meta = Metadata.fromJson(treeRow("metadata").toString)
      val treeNodesData = dataRows.filter(_("treeID") == treeRow("treeID")).map(_("nodeData")).asInstanceOf[List[Map[String, Any]]]
      LocalDecisionTreeRegressionModel.createTree(
        meta,
        LocalData.fromMapList(treeNodesData)
      )
    }
    val weights = metadata.treesMetadata.get.column("weights").get.data.asInstanceOf[List[Double]].toArray
    val ctor = classOf[GBTClassificationModel].getDeclaredConstructor(
      classOf[String],
      classOf[Array[DecisionTreeRegressionModel]],
      classOf[Array[Double]],
      classOf[Int]
    )
    ctor.setAccessible(true)
    val inst = ctor
      .newInstance(
        metadata.uid,
        trees.to[Array],
        weights,
        metadata.numFeatures.get.asInstanceOf[java.lang.Integer]
      )
    inst
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
    inst
  }

  override implicit def getTransformer(transformer: GBTClassificationModel): LocalTransformer[GBTClassificationModel] = {
    new LocalGBTClassificationModel(transformer)
  }
}
