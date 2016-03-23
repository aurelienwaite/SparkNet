package apps

import java.io._

import caffe.Caffe._
import caffe.Caffe.SolverParameter.SolverMode.CPU

import com.google.protobuf.TextFormat

import libs._
import libs.RichWeights._
import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg
import org.apache.spark.sql.Row


import scala.annotation.tailrec
import scala.util.Random


object NPLMApp {
  val trainBatchSize = 64
  val testBatchSize = 1
  val channels = 1
  val height = 1
  // After model parameters are computed, we need to redestribute and add them together so as not
  // to overwhelm the driver. This number tells the driver how many submodels to use
  val coalescedModels = 10
  // compute perplexity after set number of sync intervals
  val perplexityInterval = 10

  def time(f: => Unit)={
    val s = System.currentTimeMillis
    f
    (System.currentTimeMillis - s) / 1000
  }

  def rowToArrays(in: Seq[Row]) = {
    val toVecs = in.map { r =>
      (r.getAs[linalg.Vector]("features"), r.getAs[Double]("label").toInt)
    }
    //Convert vectors to array representation, for fast array copy in the call backs
    toVecs.map { case (vec, label) =>
      (vec.toArray.map(_.toFloat), label)
    }
  }

  def makeMinibatches(in: Seq[Row], minibatchSize: Int) = {
    val arrays = rowToArrays(in)
    val grouped = arrays.grouped(minibatchSize)
    val batched = grouped.filter(_.size == minibatchSize).map(_.toArray)
    for (b <- batched) yield {
      val data = b.map(_._1)
      val label = b.map(_._2.toFloat)
      Minibatch(data.flatten, label)
    }
  }

  def buildSolverProto(epoch: Int, lr: Float, testIter: Int)(netProto: NetParameter)  = {
    SolverParameter.newBuilder
      .setNetParam(netProto)
      .setBaseLr(lr)
      .setLrPolicy("fixed")
      .setDisplay(epoch / 100)
      .setMaxIter(epoch)
      .setSnapshot(Int.MaxValue)
      .setSnapshotPrefix("/tmp/sparknet")
      .setSolverMode(CPU)
      .addTestIter(testIter)
      .setTestInterval(Int.MaxValue) //Need to include a test interval to avoid a segfault in libcaffe
      .build()
  }


  val nplmLevel = new Level(3500, "CaffeNPLM", 4){}
  val logger = Logger.getLogger(this.getClass)
  logger.setLevel(nplmLevel)
  def logNplm(toLog: String): Unit = logger.log(nplmLevel, toLog)


  def initialiseCaffeLibrary(
                            sparkNetHome: String,
                            netPrototext: File,
                            ngramSize: Int,
                            solverBuilder: NetParameter => SolverParameter
                            ) = {
    logNplm(sys.props.get("jna.library.path").getOrElse("jna.library.path not set"))
    val caffeLib = CaffeLibrary.Instance.get(sparkNetHome + "/build/libccaffe.so")
    //val caffeLib = CaffeLibrary.Instance.get()
    logNplm("Caffe library loaded")
    var netParameter = ProtoLoader.loadNetPrototxt(netPrototext.getAbsolutePath)
    logNplm(s"Proto loaded from ${netPrototext.getAbsolutePath}")
    netParameter = ProtoLoader.replaceDataLayers(netParameter, trainBatchSize, testBatchSize, channels, height, ngramSize)
    logNplm("Data layers replaced")
    val solverParameter = solverBuilder(netParameter)
    logNplm("Built solver")
    logNplm(TextFormat.printToString(solverParameter))
    val net = CaffeNet(caffeLib, solverParameter)
    logNplm("Caffe network loaded")
    net
  }

  def computePerplexity(testNet: CaffeNet, devSet: Seq[Minibatch]) = {
    logNplm("Computing perplexity")
    val logged = testNet.test(devSet)
    assert(logged.size > 0, "No test results")
    val newPerperplexity = Math.exp(logged(0)/devSet.size)
    logNplm(s"Perplexity: $newPerperplexity")
    newPerperplexity
  }

  case class Config(
                     numWorkers: Int = -1,
                     devSetFile: String = "",
                     trainSetFile: String = "",
                     netPrototext: File = new File("."),
                     syncInterval: Int = 50,
                     snapshotPrefix: String = "/tmp/",
                     sparkNetHomeOpt: Option[String] = CaffeNet.getSparkNetHome(),
                     startLearningRate: Float = 1.0f,
                     numEpochs: Int = 50,
                     samplePercentage: Option[Double] = None
                   )

  def main(args: Array[String]) {
    val parser = new scopt.OptionParser[Config]("SparkNet NPLM") {
      head("SparkNet NPLM", "1.0")
      opt[Int]('w', "num_workers") required() valueName("number of workers") action { (x, c) =>
        c.copy(numWorkers = x)
      }
      opt[String]('t', "train") required() valueName ("training set of n-grams") action { (x, c) =>
        c.copy(trainSetFile = x)
      } text ("Training set is a required property")
      opt[String]('d', "dev") required() valueName ("development set of n-grams") action { (x, c) =>
        c.copy(devSetFile = x)
      } text ("Development set is a required property")
      opt[File]('n', "net_prototext") required() valueName ("prototext for network") action { (x, c) =>
        c.copy(netPrototext = x)
      } text ("Net prototext is a required property")
      opt[String]('p', "snapshot_prefix") valueName ("snapshot prefix") action { (x, c) =>
        c.copy(snapshotPrefix = x)
      }
      opt[String]('s', "sparknet_home") valueName ("sparknet home") action { (x, c) =>
        c.copy(sparkNetHomeOpt = Option(x))
      }
      opt[Int]('y', "sync_interval") valueName("sync interval tau") action { (x, c) =>
        c.copy(syncInterval = x)
      }
      opt[Double]('a', "sample_percentage") valueName("percentage subset of the training data") action { (x, c) =>
        c.copy(samplePercentage = Option(x))
      }
      opt[Double]('l', "learning_rate") valueName("initial learning rate") action { (x, c) =>
        c.copy(startLearningRate = x.toFloat)
      }
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.exit(1))
    import cliConf._

    val conf = new SparkConf()
      .setAppName("CaffeNPLM")
      .set("spark.driver.maxResultSize", "15G")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryoserializer.buffer.max", "512m")
      .registerKryoClasses(Array(classOf[Weights], classOf[Blob]))
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val sparkNetHome = sparkNetHomeOpt getOrElse {
      val msg = "Cannot set SparkNet home"
      logger.log(Level.ERROR, msg)
      sys.error(msg)
    }
    logNplm(s"SparkNet home = $sparkNetHome")

    import org.apache.spark.sql.functions._

    // Prepare dev set
    val devSet = sqlContext.read.parquet(devSetFile).collect()
    val devSetMinibatches = makeMinibatches(devSet, 1).toSeq
    logNplm(s"Dev set contains ${devSet.length} records")


    // Prepare training set
    val loadedTrainSet = sqlContext.read.parquet(trainSetFile)
    val trainSet = samplePercentage.map(loadedTrainSet.sample(false, _, 11l )).getOrElse(loadedTrainSet)
    val getSize = udf((features: linalg.Vector) => features.size)
    val ngramSizes = trainSet.select(getSize(trainSet("features"))).distinct().map(n => n.getAs[Int](0)).collect
    assert(ngramSizes.size == 1, sys.error("NGrams have different history lengths: " + ngramSizes.mkString(",")))
    val ngramSize= ngramSizes(0)
    logNplm(s"Using ngrams of order ${ngramSize + 1}")
    val coalesced = trainSet.repartition(numWorkers)
    //coalesced.show()

    //Initialise the driver caffelib for testing perplexities
    val solverBuilder = buildSolverProto(0, 0, devSetMinibatches.size) _
    val testNet = initialiseCaffeLibrary(sparkNetHome, netPrototext,ngramSize, solverBuilder)

    @tailrec
    def iterateEpoch(epochWeights: Weights, epochCounter: Int, learningRate: Float, perplexity: Double): Weights = {
      /*
       Spark does not order RDDs. To create an ordering, we use a salted hash function. The salt varies by epoch
       which results for a different order by epoch
       */
      val salt = Random.nextInt()
      logNplm(s"Sorting data with salt $salt")
      val getSaltedHash = udf((features: linalg.Vector, label: Int) => Vector((Seq(salt, label) ++ features.toArray.map(_.toInt)) :_* ).hashCode())
      val resorted = coalesced.orderBy(getSaltedHash(trainSet("features"), trainSet("label"))).repartition(numWorkers)

      logNplm(f"Begin epoch with learning rate $learningRate%.4f and perplexity $perplexity%.4f")
      logNplm("Creating minibatches")
      val minibatched = resorted.mapPartitions { iter =>
        val shuffled = Random.shuffle(iter.toVector)
        makeMinibatches(shuffled, trainBatchSize)
      }.cache()

      val trainPartitionSizes = minibatched.mapPartitions(iter => Iterator(iter.size)).collect()
      for((p, i)<- trainPartitionSizes.view.zipWithIndex) {
        logNplm(s"Partition $i has $p minibatches")
      }
      val epoch = trainPartitionSizes.min
      val solverBuilder = buildSolverProto(epoch, learningRate, devSet.size) _
      @transient lazy val net = initialiseCaffeLibrary(sparkNetHome, netPrototext, ngramSize, solverBuilder)
      val numIterations = epoch / syncInterval

      @tailrec
      def iterate(netWeights: Weights, iterationCounter: Int): Weights = {
        val i = numIterations - iterationCounter
        if(i % perplexityInterval == 0 ) {
          val perplexity = computePerplexity(net, devSetMinibatches)
          logNplm(s"For iteration $i perplexity: $perplexity")
        }
        logNplm("broadcasting weights")
        val broadcastWeights = sc.broadcast(netWeights)
        logNplm("training")

        val trained = minibatched.mapPartitions { trainMinibatchIt =>
            assert(trainMinibatchIt.hasNext)
            logNplm("setting weights on worker")
            net.setWeights(broadcastWeights.value)
            logNplm("running minibatches")
            val toTrain = trainMinibatchIt.drop(syncInterval * i).take(syncInterval).toSeq
            val elapsed = time {
              net.train(toTrain)
            }
            Iterator((net.getWeights(), elapsed))
        }
        logNplm("collecting weights")
        val (updatedWeights, elapsed) = trained.repartition(coalescedModels).reduce{
          case ((aWeights, aElapsed), (bWeights, bElapsed)) => (aWeights add bWeights, Math.max(aElapsed, bElapsed))
        }
        logNplm(s"caffe library completed in $elapsed seconds")
        updatedWeights.scalarDivide(1F * numWorkers)
        if (iterationCounter == 1)
          updatedWeights
        else
          iterate(updatedWeights, iterationCounter - 1)
      }
      val optimisedWeights = iterate(epochWeights, numIterations)
      minibatched.unpersist()
      testNet.setWeights(optimisedWeights)
      val snapshotPath = s"$snapshotPrefix/sparknet_epoch_${numEpochs - epochCounter}"
      logNplm(s"saving weights to $snapshotPath")
      testNet.saveWeightsToFile(snapshotPath)
      val newPerplexity = computePerplexity(testNet, devSetMinibatches)

      val (updatedLr, updatedWeights, updatedPerplexity) = if(newPerplexity > perplexity){
        logNplm("Halving learning rate")
        (learningRate/2f, optimisedWeights, newPerplexity)
      }
      else
        (learningRate, optimisedWeights, newPerplexity)
      if(epochCounter == 1)
        return updatedWeights
      else
        iterateEpoch(updatedWeights, epochCounter -1, updatedLr, updatedPerplexity)
    }
    val startWeights = testNet.getWeights()
    val weights = iterateEpoch(startWeights, numEpochs, startLearningRate, Double.PositiveInfinity)
    logNplm("finished training")
  }
}
