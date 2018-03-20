package io.hydrosphere.spark_ml_serving

import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors

class LocalModelSpec22 extends GenericTestSpec {

  modelTest(
    data = session.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label"),
    steps = Seq(
      new Tokenizer().setInputCol("text").setOutputCol("words"),
      new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
      new LogisticRegression().setMaxIter(10).setRegParam(0.01)
    ),
    columns = Seq(
      "prediction"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text"),
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
    data = session.createDataFrame(Seq(
      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
    )).toDF("features", "label"),
    steps = Seq(
      new LinearSVC()
        .setMaxIter(10)
        .setRegParam(0.1)
    ),
    columns = Seq(
      "prediction"
    )
  )

}
