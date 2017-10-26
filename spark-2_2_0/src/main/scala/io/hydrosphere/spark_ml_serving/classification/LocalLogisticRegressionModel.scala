package io.hydrosphere.spark_ml_serving.classification

import java.lang.Boolean

import io.hydrosphere.spark_ml_serving.common._
import io.hydrosphere.spark_ml_serving.common.classification.LocalProbabilisticClassificationModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.{Matrix, SparseMatrix, Vector, Vectors}

class LocalLogisticRegressionModel(override val sparkTransformer: LogisticRegressionModel)
  extends LocalProbabilisticClassificationModel[LogisticRegressionModel] {

}

object LocalLogisticRegressionModel extends LocalModel[LogisticRegressionModel] {
  override def load(metadata: Metadata, data: LocalData): LogisticRegressionModel = {
    val constructor = classOf[LogisticRegressionModel].getDeclaredConstructor(classOf[String], classOf[Matrix], classOf[Vector], classOf[Int], java.lang.Boolean.TYPE)
    constructor.setAccessible(true)
    val coefficientMatrixParams = data.column("coefficientMatrix").get.data.head.asInstanceOf[Map[String, Any]]
    val coefficientMatrix = new SparseMatrix(
      coefficientMatrixParams("numRows").asInstanceOf[Int],
      coefficientMatrixParams("numCols").asInstanceOf[Int],
      coefficientMatrixParams("colPtrs").asInstanceOf[List[Int]].toArray[Int],
      coefficientMatrixParams("rowIndices").asInstanceOf[List[Int]].toArray[Int],
      coefficientMatrixParams("values").asInstanceOf[List[Double]].toArray[Double],
      coefficientMatrixParams("isTransposed").asInstanceOf[Boolean]
    )
    val interceptVectorParams = data.column("interceptVector").get.data.head.asInstanceOf[Map[String, Any]]
    val interceptVector = Vectors.dense(interceptVectorParams("values").asInstanceOf[List[Double]].toArray[Double])
    constructor
      .newInstance(
        metadata.uid,
        coefficientMatrix,
        interceptVector,
        data.column("numFeatures").get.data.head.asInstanceOf[java.lang.Integer],
        data.column("isMultinomial").get.data.head.asInstanceOf[java.lang.Boolean]
      )
      .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
      .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
      .setProbabilityCol(metadata.paramMap("probabilityCol").asInstanceOf[String])
      .setThreshold(metadata.paramMap("threshold").asInstanceOf[Double])
  }

  override implicit def getTransformer(transformer: LogisticRegressionModel): LocalTransformer[LogisticRegressionModel] = new LocalLogisticRegressionModel(transformer)
}
