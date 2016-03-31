package com.sdl.nplm

import java.io._

import caffe.Caffe._
import caffe.Caffe.SolverParameter.SolverMode.CPU
import com.google.protobuf.TextFormat
import com.sdl.caffe._
import com.sdl.caffe.ProtoLoader
import com.sdl.caffe.RichWeights._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext, SparkEnv}
import org.apache.spark.mllib.linalg
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel

import scala.annotation.tailrec
import scala.concurrent.duration._
import scala.util.Random


object NPLMApp {
  val trainBatchSize = 64
  //TODO: Should be settable. Need some logic to pad with zeros
  val testBatchSize = 12
  val channels = 1
  val height = 1
  // After model parameters are computed, we need to redistribute and add them together so as not
  // to overwhelm the driver. This number tells the driver how many submodels to use
  val coalescedModels = 10
  // compute perplexity after set number of sync intervals
  val perplexityInterval = 10
  //Scala can't determine the implicit for the repartition ordering. Specify exactly
  def repartitionOrdering = new Ordering[(Weights, Duration, Set[String])]{
    override def compare(x: (Weights, Duration, Set[String]), y: (Weights, Duration, Set[String])) =
      weightOrdering.compare(x._1, y._1)
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

  def buildSolverProto(epoch: Int, lr: Float, testIter: Int, netProto: NetParameter)  = {
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
                            ngramSize: Int, epoch: Int, lr: Float, testIter: Int, trainBatchSize: Int,
                            testBatchSize: Int
                            ) = {
    logNplm(sys.props.get("jna.library.path").getOrElse("jna.library.path not set"))
    val libLocation = sparkNetHome + "/build/libccaffe.so"
    var netParameter = ProtoLoader.loadNetPrototxt(netPrototext.getAbsolutePath, libLocation)
    logNplm(s"Proto loaded from ${netPrototext.getAbsolutePath}")
    netParameter = ProtoLoader.replaceDataLayers(netParameter, trainBatchSize, testBatchSize, channels, height, ngramSize)
    logNplm("Data layers replaced")
    val solverParameter = buildSolverProto(epoch, lr, testIter, netParameter)
    logNplm("Built solver")
    logNplm(TextFormat.printToString(solverParameter))
    new CaffeNet(libLocation, solverParameter.toByteArray)
  }

  def computePerplexity(testNet: CaffeNet, devSet: Seq[Minibatch]) = {
    logNplm("Computing perplexity")
    val logged = testNet.test(devSet)
    assert(logged.size > 0, "No test results")
    val newPerperplexity = Math.exp(logged(0)/devSet.length)
    logNplm(s"Perplexity: $newPerperplexity")
    newPerperplexity
  }

  def computePerplexity(testNets: RDD[CaffeNet], devSet: RDD[Minibatch], devSetLength: Int) = {
    logNplm("Computing perplexity")
    val logged = testNets.zipPartitions(devSet) { (iterN, iterM) =>
      assert(iterN.hasNext, "No neural network!")
      for(testNet <- iterN) yield
        testNet.test(iterM.toSeq)
    }.reduce{ (scoresA, scoresB) => for((a,b) <- scoresA zip scoresB) yield a + b}
    assert(logged.size > 0, "No test results")
    val newPerperplexity = Math.exp(logged(0)/devSetLength)
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
                     samplePercentage: Option[Double] = None,
                     maxWorkerTime: Option[Duration] = None
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
      opt[Long]('m', "max_worker_time") valueName("Stop workers that continue after this time (minutes)") action { (x, c) =>
        c.copy(maxWorkerTime = Option(x.minutes))
      }
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.exit(1))
    import cliConf._

    val conf = new SparkConf()
      .setAppName("CaffeNPLM")
      .setIfMissing("spark.driver.maxResultSize", "15G")
      .setIfMissing("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .setIfMissing("spark.kryoserializer.buffer.max", "512m")
      .setIfMissing("spark.rdd.compress", "true")
      .setIfMissing("spark.broadcast.compress", "true")
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
    val devSetMinibatches = makeMinibatches(devSet, testBatchSize).toVector
    logNplm(s"Dev set contains ${devSet.length} records")
    /*
     * When initialising the weights on workers, we check that the weights are in sync. We do this
     * by computing the perplexity on a small amount of dev data
     */
    val miniDevSet = devSetMinibatches.take(100)
    val devSetRDD = sc.parallelize(devSetMinibatches).repartition(numWorkers).cache

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

    @tailrec
    def iterateEpoch(epochWeights: Option[Weights], epochCounter: Int, learningRate: Float, perplexity: Double): Weights = {
      /*
       Spark does not order RDDs. To create an ordering, we use a salted hash function. The salt varies by epoch
       which results for a different order by epoch
       */
      //val salt = Random.nextInt()
      //logNplm(s"Sorting data with salt $salt")
      //val getSaltedHash = udf((features: linalg.Vector, label: Int) => Vector((Seq(salt, label) ++ features.toArray.map(_.toInt)) :_* ).hashCode())
      //val resorted = coalesced.orderBy(getSaltedHash(trainSet("features"), trainSet("label"))).repartition(numWorkers)

      logNplm(f"Begin epoch with learning rate $learningRate%.4f and perplexity $perplexity%.4f")
      logNplm("Creating minibatches")
      val minibatched = coalesced.mapPartitions { iter =>
        val shuffled = Random.shuffle(iter.toVector)
        makeMinibatches(shuffled, trainBatchSize)
      }.cache()

      val trainPartitionSizes = minibatched.mapPartitions(iter => Iterator(iter.size)).collect()
      for((p, i)<- trainPartitionSizes.view.zipWithIndex) {
        logNplm(s"Partition $i has $p minibatches")
      }
      val epoch = trainPartitionSizes.min

      val driverNet = initialiseCaffeLibrary(sparkNetHome, netPrototext, ngramSize, epoch, learningRate,
        devSetMinibatches.size, trainBatchSize, testBatchSize)
      epochWeights.map(driverNet.setWeights(_))
      // Initialise the nets. Using a RDD in this way means that if a worker fails, then it will retrieve the latest
      // weights from the driver.
      logNplm("Distributing nets")
      val nets = minibatched.mapPartitions(_ => Iterator(driverNet)).cache()
      //Debugging to make sure all nets are synced correctly
      val initialPerplexities = nets.map{ net =>
         Set(computePerplexity(net, miniDevSet))
      }.reduce(_ union _)
      logNplm(s"Initial perplexities are: $initialPerplexities")
      assert(initialPerplexities.size == 1, "Nets are not correctly synced")

      val numIterations = epoch / syncInterval

      @tailrec
      def iterate(update: Weights, iterationCounter: Int): Weights = {
        val i = numIterations - iterationCounter
        if(i % perplexityInterval == 0 ) {
          val perplexity = computePerplexity(nets, devSetRDD, devSetMinibatches.length)
          logNplm(s"For iteration $i perplexity: $perplexity")
        }
        logNplm("broadcasting weights")
        val broadcastWeights = sc.broadcast(update)
        logNplm("training")
        val trained = nets.zipPartitions(minibatched){ (netIt, trainMinibatchIt) =>
          assert(netIt.hasNext, "No network for partition")
          val net = netIt.next()
          logNplm("setting weights on worker")
          val updatedWeights = net.getWeights() add broadcastWeights.value
          net.setWeights(updatedWeights)
          logNplm("running minibatches")
          assert(trainMinibatchIt.hasNext, "Run out of minibatches")
          val toTrain = trainMinibatchIt.drop(syncInterval * i).take(syncInterval).toSeq
          val (allTrained, elapsed) = net.train(toTrain, maxWorkerTime)
          val executorId = if(allTrained) Set.empty[String] else Set(SparkEnv.get.executorId)
          val diff = net.getWeights() subtract updatedWeights
          net.setWeights(updatedWeights)
          Iterator((diff, elapsed, executorId))
        }.persist(StorageLevel.MEMORY_ONLY_SER)
        logNplm("collecting weights")
        def reducer(a:(Weights, Duration, Set[String]), b:(Weights, Duration, Set[String])) =
          (a._1 add b._1, a._2 max b._2, a._3 union b._3)
        val repartitioned = trained.repartition(coalescedModels)(repartitionOrdering)
          .persist(StorageLevel.MEMORY_ONLY_SER)
        val reduced = repartitioned.mapPartitions{ iter =>
          Iterator(iter.reduce(reducer(_,_)))
        }.persist(StorageLevel.MEMORY_ONLY_SER)
        val (diff, elapsed, terminatedExecutors) = reduced.reduce(reducer(_,_))
        logNplm(s"caffe library completed in ${elapsed.toMinutes} minutes")
        logNplm(s"The following executors were took longer than the max time: $terminatedExecutors")
        diff.scalarDivide(numWorkers.toFloat)
        driverNet.setWeights(driverNet.getWeights() add diff)
        if (iterationCounter == 1)
          driverNet.getWeights()
        else
          iterate(diff, iterationCounter - 1)
      }
      val optimisedWeights = iterate(driverNet.getWeights() subtract driverNet.getWeights(), numIterations)
      minibatched.unpersist()
      nets.unpersist()
      driverNet.setWeights(optimisedWeights)
      val snapshotPath = s"$snapshotPrefix/sparknet_epoch_${numEpochs - epochCounter}"
      logNplm(s"saving weights to $snapshotPath")
      driverNet.saveWeightsToFile(snapshotPath)
      val newPerplexity = computePerplexity(nets, devSetRDD, devSetMinibatches.length)
      val (updatedLr, updatedWeights, updatedPerplexity) = if(newPerplexity > perplexity){
        logNplm("halving learning rate")
        (learningRate/2f, optimisedWeights, newPerplexity)
      }
      else
        (learningRate, optimisedWeights, newPerplexity)
      if(epochCounter == 1)
        return updatedWeights
      else
        iterateEpoch(Option(updatedWeights), epochCounter -1, updatedLr, updatedPerplexity)
    }
    val weights = iterateEpoch(None, numEpochs, startLearningRate, Double.PositiveInfinity)
    logNplm("finished training")
  }
}
