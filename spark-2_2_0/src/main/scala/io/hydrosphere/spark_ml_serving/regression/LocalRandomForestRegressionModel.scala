package io.hydrosphere.spark_ml_serving.regression

import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, RandomForestRegressionModel}

/**
  * Created by Bulat on 13.04.2017.
  */
class LocalRandomForestRegressionModel(override val sparkTransformer: RandomForestRegressionModel)
  extends LocalPredictionModel[RandomForestRegressionModel] {

}

object LocalRandomForestRegressionModel extends LocalModel[RandomForestRegressionModel] {
  override def load(metadata: Metadata, data: Map[String, Any]): RandomForestRegressionModel = {
    val treesMetadata = metadata.paramMap("treesMetadata").asInstanceOf[Map[String, Any]]
    val trees = treesMetadata map { treeKv =>
      val treeMeta = treeKv._2.asInstanceOf[Map[String, Any]]
      val meta = treeMeta("metadata").asInstanceOf[Metadata]
      LocalDecisionTreeRegressionModel.createTree(
        meta,
        data(treeKv._1).asInstanceOf[Map[String, Any]]
      )
    }
    val ctor = classOf[RandomForestRegressionModel].getDeclaredConstructor(classOf[String], classOf[Array[DecisionTreeRegressionModel]], classOf[Int])
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

  override implicit def getTransformer(transformer: RandomForestRegressionModel): LocalTransformer[RandomForestRegressionModel] = new LocalRandomForestRegressionModel(transformer)
}