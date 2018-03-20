package io.hydrosphere.spark_ml_serving

import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._

class LocalModelSpec21 extends GenericTestSpec {

  modelTest(
    data = session
      .createDataFrame(
        Seq(
          "Hi I heard about Spark".split(" "),
          "I wish Java could use case classes".split(" "),
          "Logistic regression models are neat".split(" ")
        ).map(Tuple1.apply)
      )
      .toDF("text"),
    steps = Seq(
      new Word2Vec()
        .setInputCol("text")
        .setOutputCol("result")
        .setVectorSize(3)
        .setMinCount(0)
    ),
    columns = Seq(
      "result"
    )
  )

  modelTest(
    data = session
      .createDataFrame(
        Seq(
          (0L, "a b c d e spark", 1.0),
          (1L, "b d", 0.0),
          (2L, "spark f g h", 1.0),
          (3L, "hadoop mapreduce", 0.0)
        )
      )
      .toDF("id", "text", "label"),
    steps = Seq(
      new Tokenizer().setInputCol("text").setOutputCol("words"),
      new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
      new LogisticRegression().setMaxIter(10).setRegParam(0.01)
    ),
    columns = Seq(
      "prediction"
    )
  )

}
