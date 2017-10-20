package io.hydrosphere.spark_ml_serving.common

import scala.reflect.runtime.{universe => ru}

case class LocalDataColumn[T: ru.TypeTag](name: String, data: List[T]) {
  def mapAs[A: ru.TypeTag, R](func: A => R) = {
    data match {
      case x: List[A] if ru.typeOf[T] <:< ru.typeOf[A] => x.asInstanceOf[List[A]].map(func)
      case _ => throw new IllegalArgumentException(s"Column $name is not typed as List[Double]")
    }
  }
}


class LocalData(private val columnData: List[LocalDataColumn[_]]) {

  def withColumn(localDataColumn: LocalDataColumn[_]): LocalData = {
    LocalData(columnData :+ localDataColumn)
  }

  def column(columnName: String): Option[LocalDataColumn[_]] = {
    columnData.find(_.name == columnName)
  }

  def columnNames: List[String] = columnData.map(_.name)

  def select(names: String*): LocalData = {
    LocalData(columnData.filter(column => names.contains(column.name)))
  }

  def toMapList: List[Map[String, _]] = {
    val rowCount = (for (column <- columnData) yield column.data.length).max

    for (rowNumber <- List.range(0, rowCount))
      yield (for (column <- columnData) yield column.name -> column.data(rowNumber)).toMap
  }

  def toListMap: Map[String, List[Any]] = {
    columnData.map{col =>
      col.name -> col.data
    }.toMap
  }

  override def toString: String = {

    def rowSeparator(colSizes: List[Int]): String = {
      colSizes.map("-" * _).mkString("+", "+", "+")
    }

    def rowFormat(items: List[String], colSizes: List[Int]): String = {
      items.zip(colSizes).map((t) => if (t._2 == 0) "" else s"%${t._2}s".format(t._1)).mkString("|", "|", "|")
    }

    var stringParts = List.empty[String]

    val rowCount = (for (column <- columnData) yield column.data.length).max
    val sizes = columnData.map(column => (List(column.name) ++ column.data.map(_.toString)).map(_.length).max + 1)

    stringParts :+= rowSeparator(sizes)
    stringParts :+= rowFormat(columnNames, sizes)
    stringParts :+= rowSeparator(sizes)
    for (rowNumber <- List.range(0, rowCount)) {
      val row = columnData.map { (column) =>
        if (column.data.length <= rowNumber) {
          "–"
        } else {
          column.data(rowNumber).toString
        }
      }

      stringParts :+= rowFormat(row, sizes)
    }
    stringParts :+= rowSeparator(sizes)

    stringParts.mkString("\n")
  }
}

object LocalData {

  def apply(columns: LocalDataColumn[_]*): LocalData = {
    new LocalData(columns.toList)
  }

  def apply(columns: List[LocalDataColumn[_]]): LocalData = new LocalData(columns)

}