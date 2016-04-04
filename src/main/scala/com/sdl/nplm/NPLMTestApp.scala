package com.sdl.nplm

import java.io.{File, FileOutputStream}

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.Output
import org.apache.spark.{SparkConf, SparkContext}
import com.sdl.caffe.WeightOps

/**
  * Created by rorywaite on 01/03/2016.
  */
object NPLMTestApp {

  def main(args: Array[String]) {

    import NPLMApp._

    val conf = new SparkConf()
      .setAppName("Test!")
      .setMaster("local[1]")
      .set("spark.driver.maxResultSize", "5G")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    /*val data = sqlContext.read.parquet("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/models/lm.nplm-samples.s0.t5.v20000/train.parquet")
    val sampled = rowToArrays(data.sample(false, 0.1, 11l ).collect())
    for((f, l) <- sampled) {
      println(f.map(_.toInt).mkString(" ") +s" $l")
    }
    sys.exit()*/

    val devSet = sqlContext.read.parquet("/misc/home/rwaite/mt-work/exps/WMT15_ENDE_NN/models/dev.parquet").collect()
    val minibatches = makeMinibatches(devSet,12).toSeq

    val trainBatches = makeMinibatches(devSet, 500).toSet
    val testNet = initialiseCaffeLibrary("/misc/home/rwaite/mt-software/SparkNet",
      new File("/misc/home/rwaite/mt-work/exps/WMT15_ENDE_NN/spark_net/nplm_prob.conf"),
      4, 1, 1, minibatches.size, 500,12)
    val randomWeights = testNet.getWeights()
    //testNet.loadWeightsFromFile("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/good.caffemodel")
    //println("weights loaded")
    //computePerplexity(testNet, minibatches)
    //val weights = testNet.getWeights()
    //testNet.loadWeightsFromFile("/misc/home/rwaite/mt-work/exps/G0013.NN/translation.model.v2/spark_net/caffe_snapshot/sparknet_epoch_45")
    //testNet.setWeights(randomWeights)
    //computePerplexity(testNet, minibatches)
    //testNet.setWeights(weights)
    //computePerplexity(testNet, minibatches)
    /*println("Testing ops")  */
    val kryo = new Kryo();
    val zeros = WeightOps.subtract(randomWeights,randomWeights)
    val outZeros = new Output(new FileOutputStream("/tmp/zeros.bin"));
    kryo.writeObject(outZeros, zeros)
    outZeros.close()
    /*println("testing summed")
    val summed =   WeightOps.diffAdd(zeros, WeightOps.scalarDivide(zeros,1))
    val newWeights = WeightOps.netAdd(randomWeights, summed)
    testNet.setWeights(newWeights)            */
    val toTrain = trainBatches.take(10).toSeq
    println(s"Hello! ${toTrain.size}")
    testNet.train(toTrain, None)
    val trained = testNet.getWeights()
    testNet.setWeights(trained)
    //computePerplexity(testNet, minibatches)
    println("starting subtract")
    val diff = WeightOps.subtract(trained, randomWeights)
    val output = new Output(new FileOutputStream("/tmp/diff_large_vocab.bin"));
    kryo.writeObject(output, diff)
    output.close()

    //println(diff)
  }
}
