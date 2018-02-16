package io.hydrosphere.spark_ml_serving.regression

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, RandomForestRegressionModel}

/**
  * Created by Bulat on 13.04.2017.
  */
class LocalRandomForestRegressionModel(override val sparkTransformer: RandomForestRegressionModel)
  extends LocalPredictionModel[RandomForestRegressionModel] {}

object LocalRandomForestRegressionModel
  extends TreeModelLoader[RandomForestRegressionModel]
  with TypedTransformerConverter[RandomForestRegressionModel] {

  override def build(
    metadata: Metadata,
    data: LocalData,
    treeData: LocalData
  ): RandomForestRegressionModel = {
    val dataRows      = data.toMapList
    val treesMetadata = treeData.toMapList
    val trees = treesMetadata map { treeRow =>
      val meta =
        Metadata.fromJson(treeRow("metadata").toString).copy(numFeatures = metadata.numFeatures)
      val treeNodesData = dataRows
        .filter(_("treeID") == treeRow("treeID"))
        .map(_("nodeData"))
        .asInstanceOf[List[Map[String, Any]]]
      LocalDecisionTreeRegressionModel.createTree(
        meta,
        LocalData.fromMapList(treeNodesData)
      )
    }
    val ctor = classOf[RandomForestRegressionModel].getDeclaredConstructor(
      classOf[String],
      classOf[Array[DecisionTreeRegressionModel]],
      classOf[Int]
    )
    ctor.setAccessible(true)
    val inst = ctor
      .newInstance(
        metadata.uid,
        trees.to[Array],
        metadata.numFeatures.get.asInstanceOf[java.lang.Integer]
      )
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])

    inst
      .set(inst.seed, metadata.paramMap("seed").toString.toLong)
      .set(inst.subsamplingRate, metadata.paramMap("subsamplingRate").toString.toDouble)
      .set(inst.impurity, metadata.paramMap("impurity").toString)
  }

  override implicit def toLocal(
    transformer: RandomForestRegressionModel
  ) =
    new LocalRandomForestRegressionModel(transformer)
}
