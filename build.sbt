lazy val sparkVersion = util.Properties.propOrElse("sparkVersion", "2.0.2")
lazy val localSparkVersion = sparkVersion.substring(0,sparkVersion.lastIndexOf(".")).replace('.', '_')
lazy val versionRegex = "(\\d+)\\.(\\d+).*".r

lazy val commonSettings = Seq(
  organization := "io.hydrosphere",
  version := "0.3.2",
  scalaVersion := "2.11.8"
)

def addSources(sparkDir: String) = {
  Seq(
    unmanagedSourceDirectories in Compile += baseDirectory.value / sparkDir / "src" / "main" / "scala",
    unmanagedSourceDirectories in Test += baseDirectory.value / sparkDir / "src" / "test" / "scala",
    unmanagedResourceDirectories in Test += baseDirectory.value / sparkDir / "src" / "test" / "resources"
  )
}

sparkVersion match {
  case versionRegex("2", "0") => addSources("spark-2_0")
  case versionRegex("2", "1") => addSources("spark-2_1")
  case versionRegex("2", "2") => addSources("spark-2_2")
}

lazy val root = project.in(file("."))
    .settings(commonSettings)
    .settings(publishSettings)
    .settings(
      name := s"spark-ml-serving-$localSparkVersion",

      libraryDependencies ++= Seq(
        "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
        "org.json4s" %% "json4s-native" % "3.2.11",
        "com.twitter" % "parquet-hadoop-bundle" % "1.6.0",
        "org.apache.parquet" % "parquet-common" % "1.7.0",
        "org.apache.parquet" % "parquet-column" % "1.7.0",
        "org.apache.parquet" % "parquet-hadoop" % "1.7.0",
        "org.apache.parquet" % "parquet-avro" % "1.7.0",
        "org.scalactic" %% "scalactic" % "3.0.3" % "test",
        "org.scalatest" %% "scalatest" % "3.0.3" % "test"
      )
    )

lazy val publishSettings = Seq(
    publishMavenStyle := true,
    publishTo := {
      val nexus = "https://oss.sonatype.org/"
      if (isSnapshot.value)
        Some("snapshots" at nexus + "content/repositories/snapshots/")
      else
        Some("releases"  at nexus + "service/local/staging/deploy/maven2/")
    },
    publishArtifact in Test := false,
    pomIncludeRepository := { _ => false },

    pomExtra := <url>https://github.com/Hydrospheredata/spark-ml-serving</url>
      <licenses>
        <license>
          <name>Apache 2.0 License</name>
          <url>https://github.com/Hydrospheredata/spark-ml-serving/blob/master/LICENSE</url>
          <distribution>repo</distribution>
        </license>
      </licenses>
      <scm>
        <url>https://github.com/Hydrospheredata/spark-ml-serving.git</url>
        <connection>https://github.com/Hydrospheredata/spark-ml-serving.git</connection>
      </scm>
      <developers>
        <developer>
          <id>mkf-simpson</id>
          <name></name>
          <url>https://github.com/mkf-simpson</url>
          <organization>Hydrosphere</organization>
          <organizationUrl>http://hydrosphere.io/</organizationUrl>
        </developer>
        <developer>
          <id>leonid133</id>
          <name>Leonid Blokhin</name>
          <url>https://github.com/leonid133</url>
          <organization>Hydrosphere</organization>
          <organizationUrl>http://hydrosphere.io/</organizationUrl>
        </developer>
        <developer>
          <id>KineticCookie</id>
          <name>Bulat Lutfullin</name>
          <url>https://github.com/KineticCookie</url>
          <organization>Hydrosphere</organization>
          <organizationUrl>http://hydrosphere.io/</organizationUrl>
        </developer>
        <developer>
          <id>dos65</id>
          <name>Vadim Chelyshov</name>
          <url>https://github.com/dos65</url>
          <organization>Hydrosphere</organization>
          <organizationUrl>http://hydrosphere.io/</organizationUrl>
        </developer>
      </developers>
)
