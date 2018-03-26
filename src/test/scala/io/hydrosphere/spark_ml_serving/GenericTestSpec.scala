package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common.LocalData
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.mllib.linalg.{Matrix => OldMatrix, Vector => OldVector}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FunSpec}

trait GenericTestSpec extends FunSpec with BeforeAndAfterAll {
  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("test")
    .set("spark.ui.enabled", "false")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()

  def modelPath(modelName: String): String = s"./target/test_models/${session.version}/$modelName"

  def test(
    name: String,
    data: => DataFrame,
    steps: => Seq[PipelineStage],
    columns: => Seq[String],
    accuracy: Double = 0.01
  ) = {
    val path = modelPath(name.toLowerCase())
    var validation = LocalData.empty
    var localPipelineModel = Option.empty[LocalPipelineModel]

    it("should train") {
      val pipeline = new Pipeline().setStages(steps.toArray)
      val pipelineModel = pipeline.fit(data)
      validation = LocalData.fromDataFrame(pipelineModel.transform(data))
      pipelineModel.write.overwrite().save(path)
    }

    it("should load local version") {
      localPipelineModel = Some(LocalPipelineModel.load(path))
      assert(localPipelineModel.isDefined)
    }

    it("should transform LocalData") {
      val localData = LocalData.fromDataFrame(data)
      val model = localPipelineModel.get
      val result = model.transform(localData)
      columns.foreach { col =>
        val resCol = result
          .column(col)
          .getOrElse(throw new IllegalArgumentException("Result column is absent"))
        val valCol = validation
          .column(col)
          .getOrElse(throw new IllegalArgumentException("Validation column is absent"))
        resCol.data.zip(valCol.data).foreach {
          case (r: Seq[Number @unchecked], v: Seq[Number @unchecked]) if r.isInstanceOf[Seq[Number]] && v.isInstanceOf[Seq[Number]] =>
            r.zip(v).foreach {
              case (ri, vi) =>
                assert(ri.doubleValue() - vi.doubleValue() <= accuracy, s"$ri - $vi > $accuracy")
            }
          case (r: Number, v: Number) =>
            assert(r.doubleValue() - v.doubleValue() <= accuracy, s"$r - $v > $accuracy")
          case (r, n) =>
            assert(r === n)
        }
        result.column(col).foreach { resData =>
          resData.data.foreach { resRow =>
            if (resRow.isInstanceOf[Seq[_]]) {
              assert(resRow.isInstanceOf[List[_]], resRow)
            } else if (resRow.isInstanceOf[Vector] || resRow.isInstanceOf[OldVector] || resRow
              .isInstanceOf[Matrix] || resRow.isInstanceOf[OldMatrix]) {
              assert(false, s"SparkML type detected. Column: $col, value: $resRow")
            }
          }
        }
      }
    }
  }

  def modelTest(
    data: => DataFrame,
    steps: => Seq[PipelineStage],
    columns: => Seq[String],
    accuracy: Double = 0.01
  ): Unit = {
    lazy val name = steps.map(_.getClass.getSimpleName).foldLeft("") {
      case ("", b) => b
      case (a, b) => a + "-" + b
    }

    describe(name) {
      test(name, data, steps, columns, accuracy)
    }
  }
}
