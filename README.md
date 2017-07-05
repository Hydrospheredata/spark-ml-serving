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
lazy val sparkMlServingDependency = RootProject(uri("git://github.com/Hydrospheredata/spark-ml-serving.git"))

project.in(file("."))
  // your project settings
  .dependsOn(sparkMlServingDependency)
```

2. Use it
```scala
import io.hydrosphere.spark_ml_serving.{LocalPipelineModel, PipelineLoader}
import LocalPipelineModel._

// ....
val model = PipelineLoader.load("PATH_TO_MODEL") // Load
val columns = List(LocalDataColumn("text", Seq("Hello!")))
val localData = LocalData(columns)
val result = model.transform(localData) // Transformed result
```

More examples of different ML models are in [tests](/src/test/scala/io/hydrosphere/spark_ml_serving/LocalModelSpec.scala).
