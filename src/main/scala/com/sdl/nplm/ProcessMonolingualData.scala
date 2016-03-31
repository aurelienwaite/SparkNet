
package com.sdl.nplm

import java.nio.charset.Charset
import java.nio.file.{Files, Paths}

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConverters._

/**
  *
  * Takes tokenized monolingual data, and converts it to numberized n-grams stored in a
  * parquet file
  *
  * Created by rorywaite on 30/03/2016.
  */
object ProcessMonolingualData {

  val(unk, start, end) = (0, 1 ,2)

  case class Config(
                     train: String = "",
                     dev: String = "",
                     outTrain: String = "",
                     outDev: String = "",
                     outWordmap: String = "",
                     vocabSize: Int = 0,
                     ngramOrder: Int = 0
                   )

  def main(args: Array[String]) = {
    val parser = new scopt.OptionParser[Config]("ProcessMonolingualData") {
      head("ProcessMonolingualData", "1.0")
      opt[String]('t', "train") required() valueName ("Monolingual training data") action { (x, c) =>
        c.copy(train = x)
      } text ("Training set is a required property")
      opt[String]('d', "dev") required() valueName ("Monolingual development data") action { (x, c) =>
        c.copy(dev = x)
      } text ("Development set is a required property")
      opt[String]('o', "out_train") required() valueName ("output train file") action { (x, c) =>
        c.copy(outTrain = x)
      } text ("Training file output name is a required property")
      opt[String]('p', "out_dev") required() valueName ("output dev file") action { (x, c) =>
        c.copy(outDev = x)
      } text ("Dev file output name is a required property")
      opt[String]('w', "out_wmap") required() valueName ("output wordmap file") action { (x, c) =>
        c.copy(outWordmap = x)
      } text ("Dev file output name is a required property")
      opt[Int]('v', "vocab_size") required() valueName ("vocabulary size") action { (x, c) =>
        c.copy(vocabSize = x)
      }text ("Vocabulary size is a required property")
      opt[Int]('n', "ngram_order") required() valueName ("ngram order") action { (x, c) =>
        c.copy(ngramOrder = x)
      }text ("ngram order is a required property")
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.error("Unable to parse command line options"))
    import cliConf._
    val conf = new SparkConf()
      .setAppName("ProcessMonolingualData")
    val sc = new SparkContext(conf)
    val trainTok = sc.textFile(train).repartition(1000).cache
    val devTok = sc.textFile(dev).cache
    val wordCountsRDD = trainTok.flatMap(line => line.trim().split("\\s")).map(word => (word, 1))
      .reduceByKey((a, b) => a + b)
    val sortedWordCounts = wordCountsRDD.sortBy(_._2, false)
    val topWords = sortedWordCounts.take(vocabSize)
    val wordMap = (for (((w, _), i) <- topWords.view.zipWithIndex) yield
      w -> (i + 3)).toMap
    val file = Paths.get(outWordmap)
    val words = topWords.toIterable.map(_._1)
    Files.write(file, words.asJava, Charset.forName("UTF-8"))
    val wordMapBC = sc.broadcast(wordMap)

    def numberize(line: String) =
      for (word <- line.trim().split("\\s")) yield
        wordMapBC.value.getOrElse(word, unk)

    def makeNgrams(numberized: Array[Int]) : Array[Array[Int]] = {
      val res = (Array.fill(ngramOrder - 1)(start) ++ numberized :+ end).sliding(ngramOrder).toArray
      res.map{n => assert(n.length == ngramOrder, s"Ngram of incorrest order: ${n.deep}")}
      res
    }
    def process(toProcess: RDD[String]) = {
      val numberized = toProcess.map(numberize).cache
      val (outOfVocab, totalWordCount) = numberized.map{ numberized =>
        (numberized.count(_ == unk), numberized.length)
      }.reduce{ (a,b) =>
        (a._1 + b._1, a._2 + b._2)
      }
      val ngrams = numberized.flatMap(makeNgrams)
      val points = ngrams.map{ n =>
        val toDoubles = n.map(_.toDouble)
        val history = linalg.Vectors.dense(toDoubles.dropRight(1))
        LabeledPoint(toDoubles.last, history)
      }
      println(s"$outOfVocab $totalWordCount")
      val oovRate = outOfVocab.toDouble / totalWordCount.toDouble
      (points, oovRate)
    }

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    def writeNgrams(toProcess: RDD[String], out: String) = {
      val (ngrams, oovRate) = process(toProcess)
      println(f"OOV rate of $oovRate%.4f for $out")
      val df = ngrams.toDF()
      df.show(1000)
      df.write.parquet(out)
    }

    writeNgrams(trainTok, outTrain)

    writeNgrams(devTok, outDev)

  }
}
