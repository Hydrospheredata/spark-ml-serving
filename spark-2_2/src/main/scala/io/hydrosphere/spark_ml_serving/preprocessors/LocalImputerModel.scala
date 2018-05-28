package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common.{LocalData, LocalTransformer, Metadata, SimpleModelLoader}
import org.apache.spark.ml.contextless_serving.ProxyImputerModel
import org.apache.spark.ml.feature.ImputerModel
import org.apache.spark.sql.DataFrame

class LocalImputerModel(val sparkTransformer: ImputerModel) extends LocalTransformer[ImputerModel] {
  val proxy = sparkTransformer match {
    case x: ProxyImputerModel => x
    case x: ImputerModel => ProxyImputerModel(x, LocalData.fromDataFrame(x.surrogateDF))
  }

  override def transform(localData: LocalData): LocalData = {
    println(proxy)
    println(localData)
    ???
  }

}


object LocalImputerModel extends SimpleModelLoader[ImputerModel] with TypedTransformerConverter[ImputerModel] {
  override def build(metadata: Metadata, data: LocalData): ImputerModel = {
    val imputerCtor = classOf[ImputerModel].getConstructor(classOf[String], classOf[DataFrame])

    val imputer = imputerCtor.newInstance(metadata.uid, null)

    imputer
      .setInputCols(metadata.getAs[Seq[String]]("inputCols").get.toArray)
      .setOutputCols(metadata.getAs[Seq[String]]("outputCols").get.toArray)
      .set(imputer.strategy, metadata.getAs[String]("strategy").get)
      .set(imputer.missingValue, metadata.getAs[Double]("missingValue").get)

    ProxyImputerModel(imputer, data): ImputerModel
  }

  override implicit def toLocal(sparkTransformer: ImputerModel): LocalTransformer[ImputerModel] = new LocalImputerModel(sparkTransformer)
}