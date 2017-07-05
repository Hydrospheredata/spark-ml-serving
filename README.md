# Spark-ml-serving

Contextless ML implementation of Spark ML.

## Proposal
To serve small ML pipelines there is no need to create `SparkContext` and use cluster-related features.
In this project we made our implementations for ML `Transformer`s. Some of them call context-independent Spark methods.

## Structure
Instead of using `DataFrame`s, we implemented simple `LocalData` class to get rid of `SparkContext`.
All `Transformer`s are rewritten to accept `LocalData`.

## How to use
1. Import this project as dependency:
```scala
scalaVersion := "2.11.8"
libraryDependencies += "io.hydrosphere" %% "spark-ml-serving" % "0.1.1"
```

2. Use it: [example](/example/src/main/scala/Main.scala)
```scala
import io.hydrosphere.spark_ml_serving._
import LocalPipelineModel._

// ....
val model = PipelineLoader.load("PATH_TO_MODEL") // Load
val columns = List(LocalDataColumn("text", Seq("Hello!")))
val localData = LocalData(columns)
val result = model.transform(localData) // Transformed result
```

More examples of different ML models are in [tests](/src/test/scala/io/hydrosphere/spark_ml_serving/LocalModelSpec.scala).
