package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common.utils.DataUtils._
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.PCAModel
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vectors}
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
    data.column("explainedVariance") match {
      case Some(ev) =>
        // NOTE: Spark >= 2
        val numRows  = pcMap.getOrElse("numRows", 0).asInstanceOf[Int]
        val numCols  = pcMap.getOrElse("numCols", 0).asInstanceOf[Int]
        val pcValues = pcMap.getOrElse("values", List()).asInstanceOf[List[Double]].toArray
        val pc       = new DenseMatrix(numRows, numCols, pcValues)

        val evValues = ev.data.head
          .asInstanceOf[Map[String, Any]]
          .getOrElse("values", List())
          .asInstanceOf[List[Double]]
          .toArray
        val explainedVariance = new DenseVector(evValues)
        constructor
          .newInstance(metadata.uid, pc, explainedVariance)
          .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
          .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
      case None =>
        // NOTE: Spark < 2
        val numRows  = pcMap.getOrElse("numRows", 0).asInstanceOf[Int]
        val numCols  = pcMap.getOrElse("numCols", 0).asInstanceOf[Int]
        val pcValues = pcMap.getOrElse("values", List()).asInstanceOf[List[Double]].toArray

        val pc = new OldDenseMatrix(numRows, numCols, pcValues)
        constructor
          .newInstance(
            metadata.uid,
            pc.asML,
            Vectors.dense(Array.empty[Double]).asInstanceOf[DenseVector]
          )
          .setInputCol(metadata.paramMap("inputCol").asInstanceOf[String])
          .setOutputCol(metadata.paramMap("outputCol").asInstanceOf[String])
    }
  }

  override implicit def toLocal(transformer: PCAModel) =
    new LocalPCAModel(transformer)
}
