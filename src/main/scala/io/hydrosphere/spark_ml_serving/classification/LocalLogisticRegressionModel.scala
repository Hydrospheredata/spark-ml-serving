package io.hydrosphere.spark_ml_serving.classification

import java.lang.Boolean

import io.hydrosphere.spark_ml_serving._
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.{Matrix, SparseMatrix, Vector, Vectors}

class LocalLogisticRegressionModel(override val sparkTransformer: LogisticRegressionModel)
  extends LocalProbabilisticClassificationModel[LogisticRegressionModel] {

}

object LocalLogisticRegressionModel extends LocalModel[LogisticRegressionModel] {
  override def load(metadata: Metadata, data: Map[String, Any]): LogisticRegressionModel = {
    if (data.contains("coefficients")) {
      // Spark 2.0
      val constructor = classOf[LogisticRegressionModel].getDeclaredConstructor(classOf[String], classOf[Vector], classOf[Double])
      constructor.setAccessible(true)
      val coefficientsParams = data("coefficients").asInstanceOf[Map[String, Any]]
      val coefficients = Vectors.sparse(
        coefficientsParams("size").asInstanceOf[Number].intValue(),
        coefficientsParams("indices").asInstanceOf[List[Int]].toArray[Int],
        coefficientsParams("values").asInstanceOf[List[Double]].toArray[Double]
      )
      constructor
        .newInstance(metadata.uid, coefficients, data("intercept").asInstanceOf[java.lang.Double])
        .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
        .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
        .setProbabilityCol(metadata.paramMap("probabilityCol").asInstanceOf[String])
        .setThreshold(metadata.paramMap("threshold").asInstanceOf[Double])
    } else if (data.contains("coefficientMatrix")) {
      // Spark 2.1
      val constructor = classOf[LogisticRegressionModel].getDeclaredConstructor(classOf[String], classOf[Matrix], classOf[Vector], classOf[Int], java.lang.Boolean.TYPE)
      constructor.setAccessible(true)
      val coefficientMatrixParams = data("coefficientMatrix").asInstanceOf[Map[String, Any]]
      val coefficientMatrix = new SparseMatrix(
        coefficientMatrixParams("numRows").asInstanceOf[Int],
        coefficientMatrixParams("numCols").asInstanceOf[Int],
        coefficientMatrixParams("colPtrs").asInstanceOf[List[Int]].toArray[Int],
        coefficientMatrixParams("rowIndices").asInstanceOf[List[Int]].toArray[Int],
        coefficientMatrixParams("values").asInstanceOf[List[Double]].toArray[Double],
        coefficientMatrixParams("isTransposed").asInstanceOf[Boolean]
      )
      val interceptVectorParams = data("interceptVector").asInstanceOf[Map[String, Any]]
      val interceptVector = Vectors.dense(interceptVectorParams("values").asInstanceOf[List[Double]].toArray[Double])
      constructor
        .newInstance(metadata.uid, coefficientMatrix, interceptVector, new Integer(data("numFeatures").asInstanceOf[Int]), new Boolean(data("isMultinomial").asInstanceOf[Boolean]))
        .setFeaturesCol(metadata.paramMap("featuresCol").asInstanceOf[String])
        .setPredictionCol(metadata.paramMap("predictionCol").asInstanceOf[String])
        .setProbabilityCol(metadata.paramMap("probabilityCol").asInstanceOf[String])
        .setThreshold(metadata.paramMap("threshold").asInstanceOf[Double])
    } else {
      throw new Exception("Unknown LogisticRegressionModel implementation")
    }
  }

  override implicit def getTransformer(transformer: LogisticRegressionModel): LocalTransformer[LogisticRegressionModel] = new LocalLogisticRegressionModel(transformer)
}
