package libs

import java.io.File
import java.net.URLDecoder

import scala.util.Random

import com.sun.jna.Pointer
import com.sun.jna.Memory
import scala.collection.mutable.Map
import scala.collection.mutable.MutableList

import caffe._
import caffe.Caffe._
import libs._

trait MakeCallbacks[DATA] {

  def makeLabelCallback(sizes : Sizes, minibatchSampler: MinibatchSampler[_, Int]): CaffeLibrary.java_callback_t = {
    return new CaffeLibrary.java_callback_t {
      import sizes._
      def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
        val currentLabelBatch = minibatchSampler.nextLabelMinibatch()
        assert(currentLabelBatch.length == batchSize)
        for (j <- 0 to batchSize - 1) {
          data.setFloat(j * dtypeSize, 1F * currentLabelBatch(j))
        }
      }
    }
  }

  def makeDataCallback(sizes : Sizes, minibatchSampler: MinibatchSampler[DATA, _],
                       preprocessing: Option[(DATA, Array[Float]) => Unit] = None) : CaffeLibrary.java_callback_t
}



class WeightCollection(val allWeights: Map[String, MutableList[NDArray]], val layerNames: List[String]) extends java.io.Serializable {
  val numLayers = layerNames.length

  def scalarDivide(v: Float) = {
    for (i <- 0 to numLayers - 1) {
      for (j <- 0 to allWeights(layerNames(i)).length - 1) {
        allWeights(layerNames(i))(j).scalarDivide(v)
      }
    }
  }
}


object WeightCollection extends java.io.Serializable {
  def add(wc1: WeightCollection, wc2: WeightCollection): WeightCollection = {
    assert(wc1.layerNames == wc2.layerNames)
    val layerNames = wc1.layerNames
    //check that the WeightCollection objects have the same shape
    for (i <- 0 to wc1.numLayers - 1) {
      assert(wc1.allWeights(layerNames(i)).length == wc2.allWeights(layerNames(i)).length)
      for (j <- 0 to wc1.allWeights(layerNames(i)).length - 1) {
        assert(wc1.allWeights(layerNames(i))(j).shape.deep == wc2.allWeights(layerNames(i))(j).shape.deep)
      }
    }
    // add the WeightCollection objects together
    var newWeights = Map[String, MutableList[NDArray]]()
    for (i <- 0 to wc1.numLayers - 1) {
      newWeights += (layerNames(i) -> MutableList())
      for (j <- 0 to wc1.allWeights(wc1.layerNames(i)).length - 1) {
        newWeights(layerNames(i)) += NDArray.plus(wc1.allWeights(layerNames(i))(j), wc2.allWeights(layerNames(i))(j))
      }
    }
    return new WeightCollection(newWeights, layerNames)
  }
}

trait Net {

  type Preprocessor[DATA] = (DATA, Array[Float]) => Unit

  def setTrainData[DATA](minibatchSampler: MinibatchSampler[DATA, Int],
                   trainPreprocessing: Option[Preprocessor[DATA]] = None)(implicit m: MakeCallbacks[DATA])

  def setTestData[DATA](minibatchSampler: MinibatchSampler[DATA, Int],
                  len: Int, testPreprocessing: Option[Preprocessor[DATA]] = None)(implicit m: MakeCallbacks[DATA])

  def train(numSteps: Int)

  def test(): Array[Float]

  def test(index: Int) : Float

  def forward()

  def backward()

  def setWeights(weights: WeightCollection)

  def getWeights(): WeightCollection

}

case class Sizes(dtypeSize: Int, intSize: Int)

class CaffeNet(state: Pointer, caffeLib: CaffeLibrary) extends Net{
  val numLayers = caffeLib.num_layers(state)
  val layerNames = List.range(0, numLayers).map(i => caffeLib.layer_name(state, i))
  val layerNumBlobs = List.range(0, numLayers).map(i => caffeLib.num_layer_weights(state, i))

  // store callbacks to save them from garbage collection
  var imageTrainCallback : Option[CaffeLibrary.java_callback_t] = None
  var labelTrainCallback : Option[CaffeLibrary.java_callback_t] = None
  var imageTestCallback : Option[CaffeLibrary.java_callback_t] = None
  var labelTestCallback : Option[CaffeLibrary.java_callback_t] = None

  val sizes = Sizes(dtypeSize = caffeLib.get_dtype_size(), intSize = caffeLib.get_int_size())
  import sizes._
  var numTestBatches = None: Option[Int]



  override def setTrainData[DATA](minibatchSampler:  MinibatchSampler[DATA, Int], trainPreprocessing: Option[Preprocessor[DATA]] = None)
                  (implicit m: MakeCallbacks[DATA])= {
    imageTrainCallback = Option(m.makeDataCallback(sizes, minibatchSampler, trainPreprocessing))
    labelTrainCallback = Option(m.makeLabelCallback(sizes, minibatchSampler))
    caffeLib.set_train_data_callback(state, 0, imageTrainCallback.get)
    caffeLib.set_train_data_callback(state, 1, labelTrainCallback.get)
  }

  override def setTestData[DATA](minibatchSampler:  MinibatchSampler[DATA, Int], numBatches: Int, testPreprocessing: Option[Preprocessor[DATA]] = None)
                              (implicit m: MakeCallbacks[DATA])= {
    numTestBatches = Some(numBatches)
    imageTestCallback = Option(m.makeDataCallback(sizes, minibatchSampler, testPreprocessing))
    labelTestCallback = Option(m.makeLabelCallback(sizes, minibatchSampler))
    caffeLib.set_test_data_callback(state, 0, imageTestCallback.get)
    caffeLib.set_test_data_callback(state, 1, labelTestCallback.get)
  }

  def setNumTestBatches(numBatches: Int) = {
    numTestBatches = Some(numBatches)
  }

  def train(numSteps: Int) = {
    caffeLib.set_mode_cpu()
    caffeLib.solver_step(state, numSteps)
  }

  private def testInit(numMinibatches: Int): Int = {
    caffeLib.set_mode_cpu()
    caffeLib.solver_test(state, numMinibatches) // you must run this before running caffeLib.num_test_scores(state)
    val numTestScores = caffeLib.num_test_scores(state)
    numTestScores
  }

  override def test(): Array[Float] = {
    val numTestScores = testInit(numTestBatches.getOrElse{sys.error("No test minibatches")})
    val testScores = new Array[Float](numTestScores)
    for (i <- 0 to numTestScores - 1) {
      testScores(i) = caffeLib.get_test_score(state, i) // for accuracy layers, this returns the average accuracy over a minibatch
    }
    print("testScores = " + testScores.deep.toString + "\n")
    return testScores
    //return Array(0)
  }

  override def test(index: Int) : Float = {
    val numTestScores = testInit(1)
    // First value is the label
    val offset = index + 1
    assert(offset < numTestScores, s"index $index is greater than the number of categories $numTestScores")
    caffeLib.get_test_score(state, offset)
  }

  override def forward() = {
    caffeLib.set_mode_cpu()
    caffeLib.forward(state)
  }

  override def backward() = {
    caffeLib.set_mode_cpu()
    caffeLib.backward(state)
  }

  override def setWeights(allWeights: WeightCollection) = {
    assert(allWeights.numLayers == numLayers)
    for (i <- 0 to numLayers - 1) {
      assert(allWeights.allWeights(layerNames(i)).length == layerNumBlobs(i)) // check that we have the correct number of weights
      for (j <- 0 to layerNumBlobs(i) - 1) {
        val blob = caffeLib.get_weight_blob(state, i, j)
        val shape = getShape(blob)
        assert(shape.deep == allWeights.allWeights(layerNames(i))(j).shape.deep) // check that weights are the correct shape
        val flatWeights = allWeights.allWeights(layerNames(i))(j).toFlat() // this allocation can be avoided
        val blob_pointer = caffeLib.get_data(blob)
        val size = shape.product
        var t = 0
        while (t < size) {
          blob_pointer.setFloat(dtypeSize * t, flatWeights(t))
          t += 1
        }
      }
    }
  }

  override def getWeights(): WeightCollection = {
    val allWeights = Map[String, MutableList[NDArray]]()
    for (i <- 0 to numLayers - 1) {
      val weightList = MutableList[NDArray]()
      for (j <- 0 to layerNumBlobs(i) - 1) {
        val blob = caffeLib.get_weight_blob(state, i, j)
        val shape = getShape(blob)
        val data = new Array[Float](shape.product)
        val blob_pointer = caffeLib.get_data(blob)
        val size = shape.product
        var t = 0
        while (t < size) {
          data(t) = blob_pointer.getFloat(dtypeSize * t)
          t += 1
        }
        weightList += NDArray(data, shape)
      }
      allWeights += (layerNames(i) -> weightList)
    }
    return new WeightCollection(allWeights, layerNames)
  }

  def getData(): Map[String, NDArray] = {
    val allData = Map[String, NDArray]()
    val numData = caffeLib.num_data_blobs(state)
    for (i <- 0 to numData - 1) {
      val name = caffeLib.get_data_blob_name(state, i)
      val blob = caffeLib.get_data_blob(state, i)
      val shape = getShape(blob)
      val data = new Array[Float](shape.product)
      val blob_pointer = caffeLib.get_data(blob)
      val size = shape.product
      var t = 0
      while (t < size) {
        data(t) = blob_pointer.getFloat(dtypeSize * t)
        t += 1
      }
      allData += (name -> NDArray(data, shape))
    }
    return allData
  }


  def loadWeightsFromFile(filename: String) {
    caffeLib.load_weights_from_file(state, filename)
  }

  def saveWeightsToFile(filename: String) {
    caffeLib.save_weights_to_file(state, filename)
  }

  private def getShape(blob: Pointer): Array[Int] = {
    val numAxes = caffeLib.get_num_axes(blob)
    val shape = new Array[Int](numAxes)
    for (k <- 0 to numAxes - 1) {
      shape(k) = caffeLib.get_axis_shape(blob, k)
    }
    return shape
  }
}

object CaffeNet {

  implicit val makeImageCallbacks = new MakeCallbacks[ByteImage] {
    override def makeDataCallback(sizes : Sizes, minibatchSampler: MinibatchSampler[ByteImage, _],
                                  preprocessing: Option[(ByteImage, Array[Float]) => Unit] = None): CaffeLibrary.java_callback_t = {
      return new CaffeLibrary.java_callback_t() {
        import sizes._
        def invoke(data: Pointer, batchSize: Int, numDims: Int, shape: Pointer) {
          val currentImageBatch = minibatchSampler.nextImageMinibatch()
          assert(currentImageBatch.length == batchSize)
          val arrayShape = new Array[Int](numDims) // figure out what shape images Caffe expects
          for (i <- 0 to numDims - 1) {
            val dim = shape.getInt(i * intSize)
            arrayShape(i) = dim
          }
          val size = arrayShape.product
          val buffer = new Array[Float](size)

          for (j <- 0 to batchSize - 1) {
            val currentImage = currentImageBatch(j)
            val processedImage = {
              if (preprocessing.isEmpty) {
                // copy the image into a float buffer, ignoring the numChannels component
                currentImage.cropInto(buffer, new Array[Int](numDims - 1), arrayShape.slice(1, numDims))
              } else {
                preprocessing.get(currentImage, buffer)
              }
            }
            data.write(j * size * dtypeSize, buffer, 0, size)
          }
        }
      }
    }
  }

  def apply(caffeLib: CaffeLibrary, solverParameter: SolverParameter): CaffeNet = {
    val caffeLib = CaffeLibrary.INSTANCE
    val state = caffeLib.create_state()
    val byteArr = solverParameter.toByteArray()
    val ptr = new Memory(byteArr.length)
    ptr.write(0, byteArr, 0, byteArr.length)
    caffeLib.load_solver_from_protobuf(state, ptr, byteArr.length)
    return new CaffeNet(state, caffeLib)
  }

  def getSparkNetHome(): Option[String]= {
    val env = sys.env.get("SPARKNET_HOME")
    if (env.isDefined) return env
    val path = classOf[Net].getProtectionDomain().getCodeSource().getLocation().getPath()
    val decodedPath = new File(URLDecoder.decode(path, "UTF-8"))
    for{
      p1 <- Option(decodedPath.getParentFile)
      p2 <- Option(p1.getParentFile)
      p3 <- Option(p2.getParent)
    } yield p3
  }

}
