package loaders

import java.io.{File, FileInputStream}

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

import scala.util.Random

/**
 * Assumes that we have a numberized n-grams on HDFS. This is then converted to a parquet file
 */
object NPLMLoader {

  case class Config(train: String = "", dev: String = "", outTrain: String = "", outDev: String = "")

  def main(args: Array[String]) = {
    val parser = new scopt.OptionParser[Config]("NPLMLoader") {
      head("NPLMLoader", "1.0")
      opt[String]('t', "train") required() valueName ("training set of n-grams") action { (x, c) =>
        c.copy(train = x)
      } text ("Training set is a required property")
      opt[String]('d', "dev") required() valueName ("development set of n-grams") action { (x, c) =>
        c.copy(dev = x)
      } text ("Development set is a required property")
      opt[String]('o', "out_train") required() valueName ("output train file") action { (x, c) =>
        c.copy(outTrain = x)
      } text ("Training file output name is a required property")
      opt[String]('p', "out_dev") required() valueName ("output dev file") action { (x, c) =>
        c.copy(outDev = x)
      } text ("Dev file output name is a required property")
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.error("Unable to parse command line options"))
    import cliConf._

    val conf = new SparkConf().setAppName("NPLMLoader")
    val sc = new SparkContext(conf)

    def writeParquetFile(inFileName: String, outFileName: String) = {
      val in = sc.textFile(inFileName)
      val parsed = in.map { line =>
        val fields = line.split("\\s").map(_.toDouble)
        val history = linalg.Vectors.dense(fields.dropRight(1))
        LabeledPoint(fields.last, history)
      }
      val sqlContext = new org.apache.spark.sql.SQLContext(sc)
      import sqlContext.implicits._
      val df = parsed.toDF()
      df.write.parquet(outFileName)
    }

    writeParquetFile(train, outTrain)
    writeParquetFile(dev, outDev)

  }
}