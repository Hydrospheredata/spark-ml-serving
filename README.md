[![Build Status](https://travis-ci.org/Hydrospheredata/spark-ml-serving.svg?branch=master)](https://travis-ci.org/Hydrospheredata/spark-ml-serving)

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
// Artifact name is depends of what version of spark are you usng for model training:
// spark 2.0.x
libraryDependencies += "io.hydrosphere" %% "spark-2_0-ml-serving" % "0.2.0"
// spark 2.1.x
libraryDependencies += "io.hydrosphere" %% "spark-2_1-ml-serving" % "0.2.0"
// spark 2.2.x
libraryDependencies += "io.hydrosphere" %% "spark-2_2-ml-serving" % "0.2.0"
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
