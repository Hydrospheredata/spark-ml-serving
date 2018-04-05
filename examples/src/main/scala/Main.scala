import io.hydrosphere.spark_ml_serving.LocalPipelineModel
import io.hydrosphere.spark_ml_serving.common.{LocalData, LocalDataColumn}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.SparkSession

object Train extends App {

  val conf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("example")
    .set("spark.ui.enabled", "false")

  val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()

  val df = session.createDataFrame(Seq(
            (0, Array("a", "b", "c")),
            (1, Array("a", "b", "b", "c", "a"))
         )).toDF("id", "words")

   val cv = new CountVectorizer()
     .setInputCol("words")
     .setOutputCol("features")
     .setVocabSize(3)
     .setMinDF(2)

   val pipeline = new Pipeline().setStages(Array(cv))

   val model = pipeline.fit(df)
   model.write.overwrite().save("../target/test_models/2.0.2/countVectorizer")
}

object Serve extends App {

  import LocalPipelineModel._

  val model = LocalPipelineModel .load("../target/test_models/2.0.2/countVectorizer")

  val data = LocalData(List(LocalDataColumn("words", List(
    List("a", "b", "d"),
    List("a", "b", "b", "b")

  ))))
  val result = model.transform(data)

  println(result)
}
