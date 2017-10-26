package io.hydrosphere.spark_ml_serving.regression

import io.hydrosphere.spark_ml_serving.common.{DataUtils, _}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.tree.Node

class LocalGBTRegressor(override val sparkTransformer: GBTRegressionModel)
  extends LocalPredictionModel[GBTRegressionModel] {
}

object LocalGBTRegressor extends LocalModel[GBTRegressionModel] {
  override def load(metadata: Metadata, data: LocalData): GBTRegressionModel = {
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

    val parent = new GBTRegressor()
      .setMaxIter(metadata.paramMap("maxIter").asInstanceOf[Number].intValue())
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setLabelCol(metadata.paramMap("labelCol").asInstanceOf[String])
      .setSeed(metadata.paramMap("seed").toString.toLong)
      .setStepSize(metadata.paramMap("stepSize").toString.toDouble)
      .setSubsamplingRate(metadata.paramMap("subsamplingRate").toString.toDouble)
      .setImpurity(metadata.paramMap("impurity").asInstanceOf[String])
      .setMaxDepth(metadata.paramMap("maxDepth").asInstanceOf[Number].intValue())
      .setMinInstancesPerNode(metadata.paramMap("minInstancesPerNode").asInstanceOf[Number].intValue())
      .setCheckpointInterval(metadata.paramMap("checkpointInterval").asInstanceOf[Number].intValue())
      .setMinInfoGain(metadata.paramMap("minInfoGain").toString.toDouble)
      .setCacheNodeIds(metadata.paramMap("cacheNodeIds").asInstanceOf[Boolean])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
      .setMaxMemoryInMB(metadata.paramMap("maxMemoryInMB").asInstanceOf[Number].intValue())
      .setMaxBins(metadata.paramMap("maxBins").asInstanceOf[Number].intValue())
      .setLossType(metadata.paramMap("lossType").asInstanceOf[String])

    val numFeatures: Int = metadata.numFeatures.getOrElse(0)

    val cstr = classOf[GBTRegressionModel].getDeclaredConstructor(
      classOf[String],
      classOf[Array[DecisionTreeRegressionModel]],
      classOf[Array[Double]],
      classOf[Int]
    )
    cstr.setAccessible(true)
    cstr.newInstance(
      metadata.uid,
      trees.toArray,
      weights,
      new java.lang.Integer(numFeatures)
    )
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
      .setParent(parent)
  }

  override implicit def getTransformer(transformer: GBTRegressionModel): LocalTransformer[GBTRegressionModel] = new LocalGBTRegressor(transformer)
}