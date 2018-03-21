package io.hydrosphere.spark_ml_serving.clustering

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.utils.{DataUtils, ParamUtils}
import org.apache.spark.ml.clustering.{LocalLDAModel => SparkLocalLDA}
import org.apache.spark.mllib.clustering.{LocalLDAModel => OldSparkLocalLDA}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.sql.SparkSession
import DataUtils._
import scala.reflect.runtime.universe

class LocalLDAModel(override val sparkTransformer: SparkLocalLDA)
  extends LocalTransformer[SparkLocalLDA] {

  lazy val oldModel: OldSparkLocalLDA = {
    val mirror     = universe.runtimeMirror(sparkTransformer.getClass.getClassLoader)
    val parentTerm = universe.typeOf[SparkLocalLDA].decl(universe.TermName("oldLocalModel")).asTerm
    mirror.reflect(sparkTransformer).reflectField(parentTerm).get.asInstanceOf[OldSparkLocalLDA]
  }

  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getFeaturesCol) match {
      case Some(column) =>
        val newData = column.data.mapToMlLibVectors.map(oldModel.topicDistribution(_).toList)
        localData.withColumn(
          LocalDataColumn(
            sparkTransformer.getTopicDistributionCol,
            newData
          )
        )
      case None => localData
    }
  }
}

object LocalLDAModel
  extends SimpleModelLoader[SparkLocalLDA]
  with TypedTransformerConverter[SparkLocalLDA] {

  override def build(metadata: Metadata, data: LocalData): SparkLocalLDA = {
    val topics = DataUtils.constructMatrix(
      data.column("topicsMatrix").get.data.head.asInstanceOf[Map[String, Any]]
    )
    val gammaShape = data.column("gammaShape").get.data.head.asInstanceOf[java.lang.Double]
    val topicConcentration =
      data.column("topicConcentration").get.data.head.asInstanceOf[java.lang.Double]
    val docConcentration = DataUtils.constructVector(
      data.column("docConcentration").get.data.head.asInstanceOf[Map[String, Any]]
    )
    val vocabSize = data.column("vocabSize").get.data.head.asInstanceOf[java.lang.Integer]

    val oldLdaCtor = classOf[OldSparkLocalLDA].getDeclaredConstructor(
      classOf[Matrix],
      classOf[Vector],
      classOf[Double],
      classOf[Double]
    )
    val oldLDA = oldLdaCtor.newInstance(
      Matrices.fromML(topics),
      Vectors.fromML(docConcentration),
      topicConcentration,
      gammaShape
    )

    val ldaCtor = classOf[SparkLocalLDA].getDeclaredConstructor(
      classOf[String],
      classOf[Int],
      classOf[OldSparkLocalLDA],
      classOf[SparkSession]
    )

    val lda = ldaCtor.newInstance(metadata.uid, vocabSize, oldLDA, null)

    ParamUtils.set(lda, lda.optimizer, metadata)
    ParamUtils.set(lda, lda.keepLastCheckpoint, metadata)
    ParamUtils.set(lda, lda.seed, metadata)
    ParamUtils.set(lda, lda.featuresCol, metadata)
    ParamUtils.set(lda, lda.learningDecay, metadata)
    ParamUtils.set(lda, lda.checkpointInterval, metadata)
    ParamUtils.set(lda, lda.learningOffset, metadata)
    ParamUtils.set(lda, lda.maxIter, metadata)
    ParamUtils.set(lda, lda.k, metadata)
    lda
  }

  override implicit def toLocal(sparkTransformer: SparkLocalLDA): LocalTransformer[SparkLocalLDA] =
    new LocalLDAModel(sparkTransformer)

}
