package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.contextless_serving.ProxyImputerModel
import org.apache.spark.ml.feature.ImputerModel
import org.apache.spark.sql.DataFrame

class LocalImputerModel(val sparkTransformer: ImputerModel) extends LocalTransformer[ImputerModel] {
  val proxy = sparkTransformer match {
    case x: ProxyImputerModel => x
    case x: ImputerModel => ProxyImputerModel(x, LocalData.fromDataFrame(x.surrogateDF))
  }

  val trueTransformer = proxy.imputerModel
  val surrogates = proxy.surrogate
  val missingVal = trueTransformer.getMissingValue

  override def transform(localData: LocalData): LocalData = {

    val result = trueTransformer.getInputCols.zip(trueTransformer.getOutputCols).toList.map {
      case (inColName, outColName) =>
        val surCol = surrogates.column(inColName).getOrElse(throw new IllegalArgumentException(s"No surrogate DF for inputCol $inColName"))
        val surValue = surCol.data.headOption
          .getOrElse(throw new IllegalArgumentException(s"No values in surrogate DF for inputCol $inColName"))
          .asInstanceOf[Double]

        localData.column(inColName).map { colData =>
          val replacedData = colData.data.map {
            case x if x == null => surValue
            case x: Double if java.lang.Double.compare(x, missingVal) == 0 => surValue
            case x: Float if java.lang.Double.compare(x, missingVal) == 0 => surValue
            case x: Double => x
            case x: Float => x
            case x => throw new IllegalArgumentException(s"inputCol $inColName should be Double or Float. Got $x")
          }
          LocalDataColumn(outColName, replacedData)
        }
    }
    localData.withColumns(result.flatten: _*)
  }

}


object LocalImputerModel extends SimpleModelLoader[ImputerModel] with TypedTransformerConverter[ImputerModel] {
  override def build(metadata: Metadata, data: LocalData): ImputerModel = {
    val imputerCtor = classOf[ImputerModel].getConstructor(classOf[String], classOf[DataFrame])

    val imputer = imputerCtor.newInstance(metadata.uid, null)

    val missingValue: Double = metadata.paramMap("missingValue") match {
      case x: String => x.toDouble
      case x: Double => x
      case x: Float => x
      case x => throw new IllegalArgumentException(s"Invalid missingValue: ${x}")
    }

    imputer
      .setInputCols(metadata.getAs[Seq[String]]("inputCols").get.toArray)
      .setOutputCols(metadata.getAs[Seq[String]]("outputCols").get.toArray)
      .set(imputer.strategy, metadata.getAs[String]("strategy").get)
      .set(imputer.missingValue, missingValue)

    ProxyImputerModel(imputer, data): ImputerModel
  }

  override implicit def toLocal(sparkTransformer: ImputerModel): LocalTransformer[ImputerModel] = new LocalImputerModel(sparkTransformer)
}