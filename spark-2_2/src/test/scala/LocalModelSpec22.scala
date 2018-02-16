package io.hydrosphere.spark_ml_serving

import io.hydrosphere.spark_ml_serving.common.{LocalData}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.clustering._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.{Matrix => OldMatrix, Vector => OldVector}
import org.apache.spark.ml.regression._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FunSpec}


class LocalModelSpec22 extends FunSpec with BeforeAndAfterAll {
  var session: SparkSession = _

  def modelPath(modelName: String): String = s"./target/test_models/spark-2_2_0/$modelName"

  def modelTest(data: => DataFrame, steps: => Seq[PipelineStage], columns: => Seq[String]): Unit = {
    lazy val name = steps.map(_.getClass.getSimpleName).foldLeft(""){
      case ("", b) => b
      case (a, b) => a + "-" + b
    }
    describe(name) {
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
          assert(result.column(col) === validation.column(col), s"'$col' column comparison")
          result.column(col).foreach { resData =>
            resData.data.foreach { resRow =>
              if (resRow.isInstanceOf[Seq[_]]) {
                assert(resRow.isInstanceOf[List[_]], resRow)
              } else if (resRow.isInstanceOf[Vector] || resRow.isInstanceOf[OldVector] || resRow.isInstanceOf[Matrix] || resRow.isInstanceOf[OldMatrix]) {
                assert(false, s"SparkML type detected. Column: $col, value: $resRow")
              }
            }
          }
        }
      }
    }
  }

  describe("StringIndexer-IndexToString") {
    var validation = LocalData.empty
    var localPipelineModel = Option.empty[LocalPipelineModel]
    val path = modelPath("indextostring")
    lazy val data = session.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )).toDF("id", "category")
    val columns = Seq("originalCategory")

    it("should train") {
      val indexer = new StringIndexer()
        .setInputCol("category")
        .setOutputCol("categoryIndex")
        .fit(data)
      val converter = new IndexToString()
        .setInputCol("categoryIndex")
        .setOutputCol("originalCategory")
        .setLabels(indexer.labels)
      val pipeline = new Pipeline().setStages(Array(indexer, converter)).fit(data)
      validation = LocalData.fromDataFrame(pipeline.transform(data))
      pipeline.write.overwrite().save(path)
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
        assert(result.column(col) === validation.column(col), s"'$col' column comparison")
        result.column(col).foreach { resData =>
          resData.data.foreach { resRow =>
            if (resRow.isInstanceOf[Seq[_]]) {
              assert(resRow.isInstanceOf[List[_]])
            } else if (resRow.isInstanceOf[Vector] || resRow.isInstanceOf[OldVector] || resRow.isInstanceOf[Matrix] || resRow.isInstanceOf[OldMatrix]) {
              assert(false, s"SparkML type detected. Column: $col, value: $resRow")
            }
          }
        }
      }
    }
  }

  modelTest(
    data = session.createDataFrame(
      Seq(
        (0.0, "Hi I heard about Spark"),
        (0.0, "I wish Java could use case classes"),
        (1.0, "Logistic regression models are neat")
      )
    ).toDF("label", "sentence"),
    steps = Seq(
      new Tokenizer().setInputCol("sentence").setOutputCol("words"),
      new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20),
      new IDF().setInputCol("rawFeatures").setOutputCol("features")
    ),
    columns = Seq(
      "features"
    )
  )

  modelTest(
    data = session.createDataFrame(
      Seq(
        (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
        (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
        (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
      )
    ).toDF("id", "features", "clicked"),
    steps = Seq(
      new ChiSqSelector().setNumTopFeatures(1).setFeaturesCol("features").setLabelCol("clicked").setOutputCol("selectedFeatures")
    ),
    columns = Seq(
      "selectedFeatures"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words"),
    steps = Seq(
      new CountVectorizer()
        .setInputCol("words")
        .setOutputCol("features")
        .setVocabSize(3)
        .setMinDF(2)
    ),
    columns = Seq(
      "features"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      (0, Array("Provectus", "is", "such", "a", "cool", "company")),
      (1, Array("Big", "data", "rules", "the", "world")),
      (2, Array("Cloud", "solutions", "are", "our", "future"))
    )).toDF("id", "words"),
    steps = Seq(
      new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
    ),
    columns = Seq(
      "ngrams"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      Vectors.dense(0.0, 10.3, 1.0, 4.0, 5.0),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    ).map(Tuple1.apply)).toDF("features"),
    steps = Seq(
      new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithStd(true)
        .setWithMean(false)
    ),
    columns = Seq(
      "scaledFeatures"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      (0, Seq("I", "saw", "the", "red", "balloon")),
      (1, Seq("Mary", "had", "a", "little", "lamb"))
    )).toDF("id", "raw"),
    steps = Seq(
      new StopWordsRemover()
        .setInputCol("raw")
        .setOutputCol("filtered")
    ),
    columns = Seq(
      "filtered"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -8.0)),
      (1, Vectors.dense(2.0, 1.0, -4.0)),
      (2, Vectors.dense(4.0, 10.0, 8.0))
    )).toDF("id", "features"),
    steps = Seq(
      new MaxAbsScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
    ),
    columns = Seq(
      "scaledFeatures"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (1, Vectors.dense(2.0, 1.1, 1.0)),
      (2, Vectors.dense(3.0, 10.1, 3.0))
    )).toDF("id", "features"),
    steps = Seq(
      new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
    ),
    columns = Seq(
      "scaledFeatures"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      (0, "a"), (1, "b"), (2, "c"),
      (3, "a"), (4, "a"), (5, "c")
    )).toDF("id", "category"),
    steps = Seq(
      new StringIndexer()
        .setInputCol("category")
        .setOutputCol("categoryIndex"),
      new OneHotEncoder()
        .setInputCol("categoryIndex")
        .setOutputCol("categoryVec")
    ),
    columns = Seq(
      "categoryVec"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    ).map(Tuple1.apply)).toDF("features"),
    steps = Seq(
      new PCA()
        .setInputCol("features")
        .setOutputCol("pcaFeatures")
        .setK(3)
    ),
    columns = Seq(
      "pcaFeatures"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.5, -1.0)),
      (1, Vectors.dense(2.0, 1.0, 1.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )).toDF("id", "features"),
    steps = Seq(
      new Normalizer()
        .setInputCol("features")
        .setOutputCol("normFeatures")
        .setP(1.0)
    ),
    columns = Seq(
      "normFeatures"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
      Vectors.dense(14.0, -2.0, -5.0, 1.0)
    ).map(Tuple1.apply)).toDF("features"),
    steps = Seq(
      new DCT()
        .setInputCol("features")
        .setOutputCol("featuresDCT")
        .setInverse(false)
    ),
    columns = Seq(
      "featuresDCT"
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
      new NaiveBayes()
    ),
    columns = Seq(
      "prediction"
    )
  )

  modelTest(
    data = session.createDataFrame(Seq(
      (0, 0.1),
      (1, 0.8),
      (2, 0.2)
    )).toDF("id", "feature"),
    steps = Seq(
      new Binarizer()
        .setInputCol("feature")
        .setOutputCol("binarized_feature")
        .setThreshold(5.0)
    ),
    columns = Seq(
      "binarized_feature"
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
    steps =
      Seq(
        new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel"),
        new VectorIndexer()
          .setInputCol("features")
          .setOutputCol("indexedFeatures")
          .setMaxCategories(4),
        new GBTClassifier()
          .setLabelCol("indexedLabel")
          .setFeaturesCol("indexedFeatures")
          .setMaxIter(10)
      ),
    columns = Seq(
      "predictedLabel"
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
      new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel"),
      new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4),
      new DecisionTreeClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("indexedFeatures")
    ),
    columns = Seq(
      "predictedLabel"
    )
  )

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
      new LinearRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)
    ),
    columns = Seq(
      "predictedLabel"
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
      new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
      new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
    ),
    columns = Seq(
      "prediction"
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
      new StringIndexer().setInputCol("label").setOutputCol("indexedLabel"),
      new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
      new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
    ),
    columns = Seq(
      "prediction"
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
      new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
      new RandomForestRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
    ),
    columns = Seq(
      "prediction"
    )
  )

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
      (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
      (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
      (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
      (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
      (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
      (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
    )).toDF("features", "label"),
    steps = Seq(
      new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .setMaxCategories(4),
      new GBTRegressor()
        .setLabelCol("label")
        .setFeaturesCol("indexedFeatures")
        .setMaxIter(10)
    ),
    columns = Seq(
      "prediction"
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
      new KMeans().setK(2).setSeed(1L)
    ),
    columns = Seq(
      "prediction"
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
      new GaussianMixture().setK(2)
    ),
    columns = Seq(
      "prediction"
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

  modelTest(
    data = session.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence"),
    steps = Seq(
      new RegexTokenizer()
        .setInputCol("sentence")
        .setOutputCol("words")
        .setPattern("\\W")
    ),
    columns = Seq(
      "words"
    )
  )

  override def beforeAll {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("test")
      .set("spark.ui.enabled", "false")

    session = SparkSession.builder().config(conf).getOrCreate()
  }

  override def afterAll: Unit = {
    session.stop()
  }
}
