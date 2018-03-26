package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._
import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils
import org.apache.spark.ml.feature.PCAModel
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrices, Vectors}
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, Matrices => OldMatrices}

class LocalPCAModel(override val sparkTransformer: PCAModel) extends LocalTransformer[PCAModel] {
  override def transform(localData: LocalData): LocalData = {
    localData.column(sparkTransformer.getInputCol) match {
      case Some(column) =>
        val pc      = OldMatrices.fromML(sparkTransformer.pc).asInstanceOf[OldDenseMatrix]
        val newData = column.data.mapToMlLibVectors.map(pc.transpose.multiply).map(_.toList)
        localData.withColumn(LocalDataColumn(sparkTransformer.getOutputCol, newData))
      case None => localData
    }
  }
}

object LocalPCAModel extends SimpleModelLoader[PCAModel] with TypedTransformerConverter[PCAModel] {

  override def build(metadata: Metadata, data: LocalData): PCAModel = {
    val constructor = classOf[PCAModel].getDeclaredConstructor(
      classOf[String],
      classOf[DenseMatrix],
      classOf[DenseVector]
    )
    constructor.setAccessible(true)
    val pcMap = data.column("pc").get.data.head.asInstanceOf[Map[String, Any]]
    val pcMat = DataUtils.constructMatrix(pcMap).asInstanceOf[DenseMatrix]
    data.column("explainedVariance") match {
      case Some(ev) =>
        // NOTE: Spark >= 2
        val evParams = ev.data.head.asInstanceOf[Map[String, Any]]
        val explainedVariance = DataUtils.constructVector(evParams).toDense

        constructor
          .newInstance(metadata.uid, pcMat, explainedVariance)
          .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
          .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
      case None =>
        // NOTE: Spark < 2
        constructor
          .newInstance(
            metadata.uid,
            pcMat,
            Vectors.dense(Array.empty[Double]).asInstanceOf[DenseVector]
          )
          .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
          .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
    }
  }

  override implicit def toLocal(transformer: PCAModel) =
    new LocalPCAModel(transformer)
}
