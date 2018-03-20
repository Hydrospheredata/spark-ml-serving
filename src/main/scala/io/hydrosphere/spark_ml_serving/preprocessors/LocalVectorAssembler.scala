package io.hydrosphere.spark_ml_serving.preprocessors

import io.hydrosphere.spark_ml_serving.TypedTransformerConverter
import io.hydrosphere.spark_ml_serving.common._
import org.apache.spark.ml.feature.VectorAssembler

import scala.collection.mutable

class LocalVectorAssembler(override val sparkTransformer: VectorAssembler)
  extends LocalTransformer[VectorAssembler] {

  override def transform(localData: LocalData): LocalData = {
    if (sparkTransformer.getInputCols.isEmpty) {
      localData
    } else {
      val co = sparkTransformer.getInputCols.toSeq.map { inName =>
        localData.column(inName) match {
          case Some(inCol) =>
            inCol.data.map {
              case number: java.lang.Number => Seq(number.doubleValue())
              case boolean: java.lang.Boolean => Seq(if (boolean) 1.0 else 0.0)
              case vector: Seq[Double] => vector
              case x => throw new IllegalArgumentException(s"LocalVectorAssembler does not support the ($x) ${x.getClass} type")
            }
          case None => throw new IllegalArgumentException(s"LocalVectorAssembler needs $inName column, which doesn't exist")
        }
      }

      val colLen = co.headOption.getOrElse(throw new IllegalArgumentException("Input data is empty")).length

      val result = mutable.ArrayBuffer.empty[Seq[Double]]
      var idx = 0
      while (idx < colLen) {
        val row = co.map { column =>
          column(idx)
        }
        result += row.flatten
        idx += 1
      }

      localData.withColumn(
        LocalDataColumn(
          sparkTransformer.getOutputCol,
          result.toList
        )
      )
    }
  }

  private def assemble(vv: Seq[Seq[Double]]): Seq[Double] = {
    vv.flatten
  }
}

object LocalVectorAssembler
  extends SimpleModelLoader[VectorAssembler]
  with TypedTransformerConverter[VectorAssembler] {

  override def build(metadata: Metadata, data: LocalData): VectorAssembler = {
    val assembler = new VectorAssembler(metadata.uid)
    assembler
      .setInputCols(metadata.getAs[Seq[String]]("inputCols").get.toArray)
      .setOutputCol(metadata.outputCol.get)
  }

  override implicit def toLocal(
    sparkTransformer: VectorAssembler
  ): LocalTransformer[VectorAssembler] = new LocalVectorAssembler(sparkTransformer)
}
