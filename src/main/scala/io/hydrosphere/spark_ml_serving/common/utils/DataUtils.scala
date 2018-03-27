package io.hydrosphere.spark_ml_serving.common.utils

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import io.hydrosphere.spark_ml_serving.common.{LocalData, Metadata}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, SparseMatrix, SparseVector, Vector, Vectors}
import org.apache.spark.ml.tree._
import org.apache.spark.mllib.linalg.{DenseVector => OldDenseVector, Matrix => OldMatrix, SparseVector => OldSparceVector, Vector => OldVector}

object DataUtils {
  implicit def mllibVectorToMlVector(v: OldSparceVector): SparseVector =
    new SparseVector(v.size, v.indices, v.values)

  implicit class PumpedListAny(val list: List[Any]) {
    def mapToMlVectors: List[DenseVector] = list.map(_.asInstanceOf[List[Double]].toMlVector)
    def mapToMlLibVectors: List[OldDenseVector] =
      list.map(_.asInstanceOf[List[Double]].toMlLibVector)
  }

  implicit class KindaListOfDoubles(val list: List[Double]) {

    /**
      * This is workaround for current JSON serialization. It places Int's to List[Double] and that causes exceptions.
      * WARNING: this method is very heavy, use only when you are not sure if list is pure, e.g. read from JSON.
      *
      * @return
      */
    def forceDoubles: List[Double] = list.asInstanceOf[List[AnyVal]] map (_.toString.toDouble)

    def toMlVector: DenseVector = new DenseVector(list.toArray)

    def toMlLibVector: OldDenseVector = new OldDenseVector(list.toArray)
  }

  implicit class PumpedMlVector(val vec: Vector) {
    def toList: List[Double] = vec.toArray.toList
  }

  implicit class PumpedMlLibVector(val vec: OldVector) {
    def toList: List[Double] = vec.toArray.toList
  }

  def convertFromMl(input: Any) = {
    input match {
      case x: Seq[_]    => x.toList
      case x: Vector    => x.toDense.values.toList
      case x: OldVector => x.toDense.values.toList
      case x: Matrix    => throw new IllegalArgumentException("Matrices are not supported yet")
      case x: OldMatrix => throw new IllegalArgumentException("Matrices are not supported yet")
      case x            => x
    }
  }

  def constructMatrix(params: Map[String, Any]): Matrix = {
    if (!params.contains("type")) {
      throw new IllegalArgumentException(s"Not a valid matrix: $params")
    }
    val matType = params("type").asInstanceOf[Int]
    val numRows = params("numRows").asInstanceOf[java.lang.Integer]
    val numCols = params("numCols").asInstanceOf[java.lang.Integer]
    val isTransposed = params("isTransposed").asInstanceOf[java.lang.Boolean]
    val values  = params("values").asInstanceOf[Seq[Double]].toArray

    matType match {
      case 0 => // sparse matrix
        val colPtrs    = params("colPtrs").asInstanceOf[Seq[Int]]
        val rowIndices = params("rowIndices").asInstanceOf[Seq[Int]]

        val ctor = classOf[SparseMatrix].getDeclaredConstructor(
          classOf[Int],
          classOf[Int],
          classOf[Array[Int]],
          classOf[Array[Int]],
          classOf[Array[Double]],
          classOf[Boolean]
        )

        ctor.newInstance(
          numRows,
          numCols,
          colPtrs.toArray,
          rowIndices.toArray,
          values,
          isTransposed
        ).asInstanceOf[Matrix]

      case 1 => // dense matrix

        val ctor = classOf[DenseMatrix].getDeclaredConstructor(
          classOf[Int],
          classOf[Int],
          classOf[Array[Double]],
          classOf[Boolean]
        )

        ctor.newInstance(
          numRows,
          numCols,
          values,
          isTransposed
        ).asInstanceOf[Matrix]
    }
  }

  def constructVector(params: Map[String, Any]): Vector = {
    if (params.contains("size")) {
      Vectors.sparse(
        params("size").asInstanceOf[Int],
        params("indices").asInstanceOf[Seq[Int]].toArray,
        params("values").asInstanceOf[Seq[Double]].toArray
      )
    } else {
      Vectors.dense(params("values").asInstanceOf[Seq[Double]].toArray)
    }
  }

  def createNode(nodeId: Int, metadata: Metadata, treeData: LocalData): Node = {
    val nodeRows = treeData.toMapList
    val nodeData = nodeRows.filter { _("id") == nodeId }.head
    val impurity = DataUtils.createImpurityCalculator(
      metadata.paramMap("impurity").asInstanceOf[String],
      nodeData("impurityStats").asInstanceOf[Seq[Double]].toArray
    )

    if (isInternalNode(nodeData)) {
      val ctor = classOf[InternalNode].getDeclaredConstructor(
        classOf[Double],
        classOf[Double],
        classOf[Double],
        classOf[Node],
        classOf[Node],
        classOf[Split],
        impurity.getClass.getSuperclass
      )
      ctor.newInstance(
        nodeData("prediction").asInstanceOf[java.lang.Double],
        nodeData("impurity").asInstanceOf[java.lang.Double],
        nodeData("gain").asInstanceOf[java.lang.Double],
        createNode(nodeData("leftChild").asInstanceOf[java.lang.Integer], metadata, treeData),
        createNode(nodeData("rightChild").asInstanceOf[java.lang.Integer], metadata, treeData),
        DataUtils.createSplit(nodeData("split").asInstanceOf[Map[String, Any]]),
        impurity
      )
    } else {
      val ctor = classOf[LeafNode].getDeclaredConstructor(
        classOf[Double],
        classOf[Double],
        impurity.getClass.getSuperclass
      )
      ctor.newInstance(
        nodeData("prediction").asInstanceOf[java.lang.Double],
        nodeData("impurity").asInstanceOf[java.lang.Double],
        impurity
      )
    }
  }

  def isInternalNode(nodeData: Map[String, Any]): Boolean =
    (nodeData("leftChild").asInstanceOf[java.lang.Integer] != -1) && (nodeData("rightChild")
      .asInstanceOf[java.lang.Integer] != -1)

  def createImpurityCalculator(impurity: String, stats: Array[Double]): Object = {
    val className = impurity match {
      case "gini"     => "org.apache.spark.mllib.tree.impurity.GiniCalculator"
      case "entropy"  => "org.apache.spark.mllib.tree.impurity.EntropyCalculator"
      case "variance" => "org.apache.spark.mllib.tree.impurity.VarianceCalculator"
      case _ =>
        throw new IllegalArgumentException(
          s"ImpurityCalculator builder did not recognize impurity type: $impurity"
        )
    }
    val ctor = Class.forName(className).getDeclaredConstructor(classOf[Array[Double]])
    ctor.setAccessible(true)
    ctor.newInstance(stats).asInstanceOf[Object]
  }

  def createSplit(data: Map[String, Any]): Split = {
    val cot = data("leftCategoriesOrThreshold").asInstanceOf[Seq[Double]]
    data("numCategories").toString.toInt match {
      case -1 =>
        val ctor = classOf[ContinuousSplit].getDeclaredConstructor(classOf[Int], classOf[Double])
        ctor.setAccessible(true)
        ctor.newInstance(
          data("featureIndex").asInstanceOf[java.lang.Integer],
          cot.head.asInstanceOf[java.lang.Double]
        )
      case x =>
        val ctor = classOf[CategoricalSplit].getDeclaredConstructor(
          classOf[Int],
          classOf[Array[Double]],
          classOf[Int]
        )
        ctor.setAccessible(true)
        ctor.newInstance(
          data("featureIndex").asInstanceOf[java.lang.Integer],
          cot.toArray,
          x.asInstanceOf[java.lang.Integer]
        )
    }
  }

  def kludgeForVectorIndexer(map: Map[String, Any]): Map[Int, Map[Double, Int]] = {
    map.map({
      case (k, v) =>
        val key   = k.toInt
        val value = v.asInstanceOf[Map[String, Int]].map(x => x._1.toDouble -> x._2)
        key -> value
    })
  }

  def asBreeze(values: Array[Double]): BV[Double] = new BDV[Double](values)

  def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }
}
