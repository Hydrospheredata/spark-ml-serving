package io.hydrosphere.spark_ml_serving.common

import java.io.{InputStreamReader, BufferedReader}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{LocalFileSystem, Path, FileSystem}

case class ModelSource(
  root: String,
  fs: FileSystem
) {

  def readFile(path: String): String = {
    val fsPath = filePath(path)
    val reader = new BufferedReader(new InputStreamReader(fs.open(fsPath)))

    val builder = new StringBuilder()
    var line: String = null
    while ({ line = reader.readLine(); line != null}) {
      builder.append(line + "\n")
    }
    builder.mkString
  }

  def findFile(dir: String, recursive: Boolean, f: String => Boolean): Option[Path] = {
    val dirPath = filePath(dir)
    if(fs.exists(dirPath) & fs.isDirectory(dirPath)) {
      val iter = fs.listFiles(dirPath, recursive)
      while (iter.hasNext) {
        val st = iter.next()
        if (st.isFile && f(st.getPath.getName)) return Some(st.getPath)
      }
      None
    } else {
      None
    }
  }

  def filePath(path: String): Path = {
    val rootPath = fs.getWorkingDirectory
    new Path(s"$root/$path")
  }

}

object ModelSource {

  def local(path: String): ModelSource = {
    ModelSource(path, FileSystem.getLocal(new Configuration()))
  }

  def hadoop(path: String, conf: Configuration): ModelSource = {
    val fs = FileSystem.get(conf)
    ModelSource(path, fs)
  }

}

