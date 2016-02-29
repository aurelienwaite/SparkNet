package apps

import java.io._
import java.net.URLDecoder
import java.util

import caffe.Caffe._
import caffe.Caffe.SolverParameter.SolverMode.CPU

import com.google.protobuf.TextFormat

import com.sun.jna.Pointer
import libs._
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

  val workerStore = new WorkerStore()

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

  def buildSolverProto(epoch: Int, lr: Float, testIter: Int)(netProto: NetParameter)  =
    SolverParameter.newBuilder
      .setNetParam(netProto)
      .setBaseLr(lr)
      .setLrPolicy("fixed")
      .setDisplay(epoch/ 100)
      .setMaxIter(epoch)
      .setSnapshot(Int.MaxValue)
      .setSnapshotPrefix("/tmp/sparknet")
      .setSolverMode(CPU)
      .addTestIter(testIter)
      .setTestInterval(Int.MaxValue) //Need to include a test interval to avoid a segfault in libcaffe
      .build()


  implicit val makeNPLMCallbacks = new MakeCallbacks[Array[Float]] {
    override def makeDataCallback(sizes : Sizes, minibatchSampler: MinibatchSampler[Array[Float], _ ],
                                  preprocessing: Option[(Array[Float], Array[Float]) => Unit] = None): CaffeLibrary.java_callback_t = {
      return new CaffeLibrary.java_callback_t() {
        import sizes._
        def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
          val currentBatch = minibatchSampler.nextImageMinibatch()
          assert(currentBatch.length == batchSize)
          val arrayShape = new Array[Int](numDims)
          for (i <- 0 to numDims - 1) {
            val dim = shape.getInt(i * intSize)
            arrayShape(i) = dim
          }
          val size = arrayShape.product
          for (j <- 0 to batchSize - 1) {
            val current = currentBatch(j)
            preprocessing.map {p =>
              val buffer = new Array[Float](size)
              p(current, buffer)
              data.write(j * size * dtypeSize, buffer, 0, size)
            } getOrElse {
              data.write(j * size * dtypeSize, current, 0, size)
            }

          }
        }
      }
    }

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
    workerStore.setNet("net", net)
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
                     numEpochs: Int = 50
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
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.exit(1))
    import cliConf._

    val conf = new SparkConf()
      .setAppName("CaffeNPLM")
      .set("spark.driver.maxResultSize", "5G")
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
    val devSet = sqlContext.read.parquet(devSetFile)
    val asArrays = rowToArrays(devSet.collect())
    val devMinibatches = asArrays.grouped(1).map(_.toArray).map(b => (b.map(_._1), b.map(_._2))).toSeq
    logNplm(s"Dev set contains ${asArrays.length} records")
    val labels = asArrays.map(_._2)

    // Prepare training set
    val trainSet = sqlContext.read.parquet(trainSetFile)
    val getSize = udf((features: linalg.Vector) => features.size)
    val ngramSizes = trainSet.select(getSize(trainSet("features"))).distinct().map(n => n.getAs[Int](0)).collect
    assert(ngramSizes.size == 1, sys.error("NGrams have different history lengths: " + ngramSizes.mkString(",")))
    val ngramSize= ngramSizes(0)
    logNplm(s"Using ngrams of order ${ngramSize + 1}")
    val coalesced = trainSet.repartition(numWorkers)
    coalesced.show()

    //Initialise the driver caffelib for testing perplexities
    val solverBuilder = buildSolverProto(0, 0, asArrays.size) _
    initialiseCaffeLibrary(sparkNetHome, netPrototext, ngramSize, solverBuilder)

    @tailrec
    def iterateEpoch(epochWeights: WeightCollection, epochCounter: Int, learningRate: Float, perplexity: Double): WeightCollection = {
      /*
       Spark does not sort order of RDDs. To create an ordering, we use a salted hash function. The salt varies by epoch
       which results for a different order by epoch
       */
      val salt = Random.nextInt()
      logNplm(s"Sorting data with salt $salt")
      val getSaltedHash = udf((features: linalg.Vector, label: Int) => Vector((Seq(salt, label) ++ features.toArray.map(_.toInt)) :_* ).hashCode())
      val resorted = coalesced.orderBy(getSaltedHash(trainSet("features"), trainSet("label"))).repartition(numWorkers)
      logNplm(s"Reordered data contains ${resorted.count()} rows")
      resorted.show

      logNplm(f"Begin epoch with learning rate $learningRate%.4f and perplexity $perplexity%.4f")
      logNplm("Creating minibatches")
      val minibatched = resorted.mapPartitions { iter =>
        workerStore.reset() // Clean up memory of workers before starting epoch
        val shuffled = Random.shuffle(iter.toVector)
        //Convert vectors to array representation, for fast array copy in the call backs
        val arrays = rowToArrays(shuffled)
        val grouped = arrays.grouped(trainBatchSize)
        val batched = grouped.filter(_.size == trainBatchSize).map(_.toArray)
        batched.map(b => (b.map(_._1), b.map(_._2)))
      }.cache()

      val numTrainMinibatches = minibatched.count()
      logNplm(s"Number of minibatches = $numTrainMinibatches")

      // Partition sizes also includes worker ID
      val trainPartitionSizes = minibatched.mapPartitionsWithIndex((w, iter) => Iterator((w, iter.size))).cache()
      val trainPartitionSizesCollected = trainPartitionSizes.collect()
      val trainPartitionSizesString = trainPartitionSizesCollected.map(p => s"Partition ${p._1} has ${p._2} minibatches").mkString("\n")
      logNplm(trainPartitionSizesString)
      val epoch = trainPartitionSizesCollected.map(_._2).min
      val solverBuilder = buildSolverProto(epoch, learningRate, asArrays.size) _
      val numIterations = epoch / syncInterval

      @tailrec
      def iterate(netWeights: WeightCollection, iterationCounter: Int): WeightCollection = {
        logNplm("broadcasting weights")
        val broadcastWeights = sc.broadcast(netWeights)
        logNplm("training")
        val trained = trainPartitionSizes.zipPartitions(minibatched)(
          (lenIt, trainMinibatchIt) => {
            assert(lenIt.hasNext && trainMinibatchIt.hasNext)
            val (w, len) = lenIt.next
            if(! workerStore.initialized) {
              logNplm("initialising library")
              initialiseCaffeLibrary(sparkNetHome, netPrototext, ngramSize, solverBuilder)
            }
            logNplm("setting weights on worker")
            workerStore.getNet("net").setWeights(broadcastWeights.value)
            assert(!lenIt.hasNext)
            logNplm("running minibatches")
            val elapsed = time {
              val minibatchSampler = new MinibatchSampler[Array[Float], Int](trainMinibatchIt, len, syncInterval,
                Option(numIterations - iterationCounter))
              workerStore.getNet("net").setTrainData(minibatchSampler, None)
              workerStore.getNet("net").train(syncInterval)
            }
            Iterator((workerStore.getNet("net").getWeights(), elapsed))
          }
        )
        logNplm("collecting weights")
        val (updatedWeights, elapsed) = trained.reduce{
          case ((aWeights, aElapsed), (bWeights, bElapsed)) => (WeightCollection.add(aWeights, bWeights), Math.max(aElapsed, bElapsed))
        }
        logNplm(s"caffe library completed in $elapsed seconds")
        updatedWeights.scalarDivide(1F * numWorkers)
        if (iterationCounter == 1)
          updatedWeights
        else
          iterate(updatedWeights, iterationCounter - 1)
      }
      val optimisedWeights = iterate(epochWeights, numIterations)
      trainPartitionSizes.unpersist()
      minibatched.unpersist()
      val testNet = workerStore.getNet("net")
      testNet.setWeights(optimisedWeights)
      val snapshotPath = s"$snapshotPrefix/sparknet_epoch_${numEpochs - epochCounter}"
      logNplm(s"saving weights to $snapshotPath")
      testNet.saveWeightsToFile(snapshotPath)
      logNplm("Computing perplexity")
      val len = asArrays.length
      val minibatchSampler = new MinibatchSampler(devMinibatches.toIterator, len, len)
      testNet.setTestData(minibatchSampler, len)
      val newPerperplexity = Perplexity.compute(labels, testNet)
      logNplm(s"Perplexity: $newPerperplexity")
      val (updatedLr, updatedWeights, updatedPerplexity) = if(newPerperplexity > perplexity){
        logNplm("Halving learning rate")
        (learningRate/2f, optimisedWeights, perplexity)
      }
      else
        (learningRate, optimisedWeights, newPerperplexity)
      if(epochCounter == 1)
        return updatedWeights
      else
        iterateEpoch(updatedWeights, epochCounter -1, updatedLr, updatedPerplexity)
    }
    val startWeights = workerStore.getNet("net").getWeights()
    val weights = iterateEpoch(startWeights, numEpochs, startLearningRate, Double.PositiveInfinity)
    logNplm("finished training")

  }
}
