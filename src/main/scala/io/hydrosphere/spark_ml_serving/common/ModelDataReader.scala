package io.hydrosphere.spark_ml_serving.common

import io.hydrosphere.spark_ml_serving.common.reader._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import parquet.format.converter.ParquetMetadataConverter.NO_FILTER
import parquet.hadoop.{ParquetFileReader, ParquetReader}
import parquet.schema.MessageType

import scala.collection.immutable.HashMap
import scala.collection.mutable

object ModelDataReader {

  def parse(source: ModelSource, path: String): LocalData = {
    source.findFile(path, true, _.endsWith(".parquet")) match {
      case Some(p) => readData(p)
      case None    => LocalData.empty
    }
  }

  private def readData(p: Path): LocalData = {
    val conf: Configuration = new Configuration()
    val metaData            = ParquetFileReader.readFooter(conf, p, NO_FILTER)
    val schema: MessageType = metaData.getFileMetaData.getSchema

    val reader = ParquetReader.builder[SimpleRecord](new SimpleReadSupport(), p.getParent).build()
    var result = LocalData.empty

    try {
      var value = reader.read()
      while (value != null) {
        val valMap = value.struct(HashMap.empty[String, Any], schema)
        result = mergeMaps(result, valMap)
        value  = reader.read()
      }
      result
    } finally {
      if (reader != null) {
        reader.close()
      }
    }
  }

  private def mergeMaps(acc: LocalData, map: HashMap[String, Any]) = {
    var result = acc
    map.foreach {
      case (k, v) => result = result.appendToColumn(k, List(v))
    }
    result
  }
}
