package io.hydrosphere.spark_ml_serving.clustering

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._
import org.apache.spark.ml.clustering.GaussianMixtureModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.stat.distribution.MultivariateGaussian

class LocalGaussianMixtureModel(override val sparkTransformer: GaussianMixtureModel)
  extends LocalTransformer[GaussianMixtureModel] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        val predictMethod = classOf[GaussianMixtureModel].getMethod("predict", classOf[Vector])
        predictMethod.setAccessible(true)
        val newColumn =
          LocalDataColumn(sparkTransformer.getPredictionCol, column.data.mapToMlVectors map {
            predictMethod.invoke(sparkTransformer, _).asInstanceOf[Int]
          })
        localData.withColumn(newColumn)
      case None => localData
    }
  }
}

object LocalGaussianMixtureModel
  extends SimpleModelLoader[GaussianMixtureModel]
  with TypedTransformerConverter[GaussianMixtureModel] {

  override def build(metadata: Metadata, data: LocalData): GaussianMixtureModel = {
    val weights     = data.column("weights").get.data.head.asInstanceOf[Seq[Double]].toArray
    val mus         = data.column("mus").get.data.head.asInstanceOf[Seq[Map[String, Any]]]
    val sigmas      = data.column("sigmas").get.data.head.asInstanceOf[Seq[Map[String, Any]]]
    val sigMatrices = sigmas.map(DataUtils.constructMatrix)
    val musVecs     = mus.map(DataUtils.constructVector)

    val gaussians = musVecs zip sigMatrices map {
      case (mu, sigma) => new MultivariateGaussian(mu, sigma)
    }

    val constructor = classOf[GaussianMixtureModel].getDeclaredConstructor(
      classOf[String],
      classOf[Array[Double]],
      classOf[Array[MultivariateGaussian]]
    )
    constructor.setAccessible(true)
    var inst = constructor.newInstance(metadata.uid, weights, gaussians.toArray)
    inst = inst.set(inst.probabilityCol, metadata.paramMap("probabilityCol").asInstanceOf[String])
    inst = inst.set(inst.featuresCol, metadata.paramMap("featuresCol").asInstanceOf[String])
    inst = inst.set(inst.predictionCol, metadata.paramMap("predictionCol").asInstanceOf[String])
    inst
  }

  override implicit def toLocal(
    transformer: GaussianMixtureModel
  ) = new LocalGaussianMixtureModel(transformer)
}
