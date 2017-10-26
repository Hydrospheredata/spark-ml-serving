package io.hydrosphere.spark_ml_serving.regression

import io.hydrosphere.spark_ml_serving.common.{DataUtils, _}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.tree.Node

class LocalGBTRegressor(override val sparkTransformer: GBTRegressionModel)
  extends LocalPredictionModel[GBTRegressionModel] {
}

object LocalGBTRegressor extends LocalModel[GBTRegressionModel] {
  override def load(metadata: Metadata, data: LocalData): GBTRegressionModel = {
    ???
//    val weights = metadata.paramMap("treesMetadata").asInstanceOf[Map[String, Any]] map { x =>
//      x._2.asInstanceOf[Map[String, Any]]("weights").toString.toDouble
//    }
//
//    val trees = metadata.paramMap("treesMetadata").asInstanceOf[Map[String, Any]] zip data map { x =>
//      val treeID = x._1._1
//      val meta = x._1._2.asInstanceOf[Map[String, Any]]("metadata").asInstanceOf[Metadata]
//      val weight = x._1._2.asInstanceOf[Map[String, Any]]("weights").toString.toDouble
//      val treeData =  x._2._2.asInstanceOf[Map[String, Any]]
//
//      createTree(treeID.toString, meta, treeData)
//    }
//
//    val parent = new GBTRegressor()
//      .setMaxIter(metadata.paramMap("maxIter").asInstanceOf[Number].intValue())
//      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
//      .setLabelCol(metadata.paramMap("labelCol").asInstanceOf[String])
//      .setSeed(metadata.paramMap("seed").toString.toLong)
//      .setStepSize(metadata.paramMap("stepSize").toString.toDouble)
//      .setSubsamplingRate(metadata.paramMap("subsamplingRate").toString.toDouble)
//      .setImpurity(metadata.paramMap("impurity").asInstanceOf[String])
//      .setMaxDepth(metadata.paramMap("maxDepth").asInstanceOf[Number].intValue())
//      .setMinInstancesPerNode(metadata.paramMap("minInstancesPerNode").asInstanceOf[Number].intValue())
//      .setCheckpointInterval(metadata.paramMap("checkpointInterval").asInstanceOf[Number].intValue())
//      .setMinInfoGain(metadata.paramMap("minInfoGain").toString.toDouble)
//      .setCacheNodeIds(metadata.paramMap("cacheNodeIds").asInstanceOf[Boolean])
//      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
//      .setMaxMemoryInMB(metadata.paramMap("maxMemoryInMB").asInstanceOf[Number].intValue())
//      .setMaxBins(metadata.paramMap("maxBins").asInstanceOf[Number].intValue())
//      .setLossType(metadata.paramMap("lossType").asInstanceOf[String])
//
//    val numFeatures: Int = metadata.numFeatures.getOrElse(0)
//
//    val cstr = classOf[GBTRegressionModel].getDeclaredConstructor(
//      classOf[String],
//      classOf[Array[DecisionTreeRegressionModel]],
//      classOf[Array[Double]],
//      classOf[Int]
//    )
//    cstr.setAccessible(true)
//    cstr.newInstance(
//      metadata.uid,
//      trees.toArray,
//      weights.toArray,
//      new java.lang.Integer(numFeatures)
//    )
//      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
//      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
//      .setParent(parent)
  }

  def createTree(uid: String, metadata: Metadata, data: LocalData): DecisionTreeRegressionModel = {
    val ctor = classOf[DecisionTreeRegressionModel].getDeclaredConstructor(classOf[String], classOf[Node], classOf[Int])
    ctor.setAccessible(true)
    val inst = ctor.newInstance(
      uid,
      DataUtils.createNode(0, metadata, data),
      metadata.numFeatures.get.asInstanceOf[java.lang.Integer]
    )
    inst.setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
  }

  override implicit def getTransformer(transformer: GBTRegressionModel): LocalTransformer[GBTRegressionModel] = new LocalGBTRegressor(transformer)
}