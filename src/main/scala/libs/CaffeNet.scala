package libs

import java.io._
import java.nio.file.{Paths, Files}

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import scala.collection.mutable.Map
import scala.collection.mutable.MutableList
import java.util.Arrays

object CaffeNet {
  def apply(netParam: NetParameter, schema: StructType, preprocessor: Preprocessor): DFCaffeNet = {
    return new DFCaffeNet(netParam, new FloatNet(netParam), schema, preprocessor)
  }

  def apply(netParam: NetParameter) = new CaffeNet(netParam, new FloatNet(netParam))

}


class CaffeNet (netParam: NetParameter, caffeNet: FloatNet) {
  protected val inputSize = netParam.input_size
  protected val batchSize = netParam.input_shape(0).dim(0).toInt

  // private val caffeNet = new FloatNet(netParam)
  private val inputRef = new Array[FloatBlob](inputSize)
  def getNet = caffeNet // TODO: For debugging

  protected val numOutputs = caffeNet.num_outputs
  private val numLayers = caffeNet.layers().size.toInt
  private val layerNames = List.range(0, numLayers).map(i => caffeNet.layers.get(i).layer_param.name.getString)
  private val numLayerBlobs = List.range(0, numLayers).map(i => caffeNet.layers.get(i).blobs().size.toInt)

  // Preallocate a buffer for data input into the net
  val inputs = new FloatBlobVector(inputSize)
  for (i <- 0 to inputSize - 1) {
    val dims = new Array[Int](netParam.input_shape(i).dim_size)
    for (j <- dims.indices) {
      dims(j) = netParam.input_shape(i).dim(j).toInt
    }
    // prevent input blobs from being GCed
    // see https://github.com/bytedeco/javacpp-presets/issues/140
    inputRef(i) = new FloatBlob(dims)
    inputs.put(i, inputRef(i))
  }
  val inputBuffer = new Array[Array[Float]](inputSize)
  val inputBufferSize = new Array[Int](inputSize)
  for (i <- 0 to inputSize - 1) {
    inputBufferSize(i) = JavaCPPUtils.getInputShape(netParam, i).drop(1).product // drop 1 to ignore batchSize
    inputBuffer(i) = new Array[Float](inputBufferSize(i) * batchSize)
  }



  def forward(dataBlobNames: List[String]): Map[String, NDArray] = {
    val tops = caffeNet.Forward(inputs)
    val outputs = Map[String, NDArray]()
    for (j <- 0 to numOutputs - 1) {
      val outputName = caffeNet.blob_names().get(caffeNet.output_blob_indices().get(j)).getString
      val top = tops.get(j)
      val shape = Array.range(0, top.num_axes).map(i => top.shape.get(i))
      val output = new Array[Float](shape.product)
      top.cpu_data().get(output, 0, shape.product)
      outputs += (outputName -> NDArray(output, shape))
    }
    for (name <- dataBlobNames) {
      val floatBlob = caffeNet.blob_by_name(name)
      if (floatBlob == null) {
        throw new IllegalArgumentException("The net does not have a layer named " + name + ".\n")
      }
      outputs += (name -> JavaCPPUtils.floatBlobToNDArray(floatBlob))
    }
    return outputs
  }

  def setMinibatch(minibatch: Array[Array[Float]]) = {
    assert(minibatch.size == inputSize, "Minibatch and input have different dims")
    for (i <- 0 until inputSize) {
      val blob = inputs.get(i)
      val buffer = blob.cpu_data()
      buffer.put(minibatch(i), 0, batchSize * inputBufferSize(i))
    }
  }

  def forwardBackward(): Unit= {
    val t = time {
      caffeNet.ForwardBackward(inputs)
    }
    println(s"ForwardBackward took ${t}s")
  }

  def getWeights(): WeightCollection = {
    val weights = Map[String, MutableList[NDArray]]()
    for (i <- 0 to numLayers - 1) {
      val weightList = MutableList[NDArray]()
      for (j <- 0 to numLayerBlobs(i) - 1) {
        val blob = caffeNet.layers().get(i).blobs().get(j)
        val shape = JavaCPPUtils.getFloatBlobShape(blob)
        val data = new Array[Float](shape.product)
        blob.cpu_data.get(data, 0, data.length)
        weightList += NDArray(data, shape)
      }
      weights += (layerNames(i) -> weightList)
    }
    return new WeightCollection(weights, layerNames)
  }

  def setWeights(weights: WeightCollection) = {
    assert(weights.numLayers == numLayers)
    for (i <- 0 to numLayers - 1) {
      for (j <- 0 to numLayerBlobs(i) - 1) {
        val blob = caffeNet.layers().get(i).blobs().get(j)
        val shape = JavaCPPUtils.getFloatBlobShape(blob)
        assert(shape.deep == weights.allWeights(layerNames(i))(j).shape.deep) // check that weights are the correct shape
        val flatWeights = weights.allWeights(layerNames(i))(j).toFlat() // this allocation can be avoided
        blob.cpu_data.put(flatWeights, 0, flatWeights.length)
      }
    }
  }

  def copyTrainedLayersFrom(filepath: String) = {
    if (!Files.exists(Paths.get(filepath))) {
      throw new IllegalArgumentException("The file " + filepath + " does not exist.\n")
    }
    caffeNet.CopyTrainedLayersFrom(filepath)
  }

  def saveWeightsToFile(filepath: String) = {
    val f = new File(filepath)
    f.getParentFile.mkdirs
    val netParam = new NetParameter()
    caffeNet.ToProto(netParam)
    WriteProtoToBinaryFile(netParam, filepath)
  }

}

class DFCaffeNet(netParam: NetParameter, caffeNet: FloatNet, schema: StructType, preprocessor: Preprocessor) extends CaffeNet(netParam, caffeNet) {

  private val transformations = new Array[Any => NDArray](inputSize)
  private val columnNames = schema.map(entry => entry.name)
  private val inputIndices = new Array[Int](inputSize)
  for (i <- 0 to inputSize - 1) {
    val name = netParam.input(i).getString
    transformations(i) = preprocessor.convert(name, JavaCPPUtils.getInputShape(netParam, i).drop(1)) // drop first index to ignore batchSize
    inputIndices(i) = columnNames.indexOf(name)
  }

  def transformInto(iterator: Iterator[Row], data: FloatBlobVector) = {
    var batchIndex = 0
    while (iterator.hasNext && batchIndex != batchSize) {
      val row = iterator.next
      for (i <- 0 to inputSize - 1) {
        val result = transformations(i)(row(inputIndices(i)))
        val flatArray = result.toFlat() // TODO: Make this efficient
        System.arraycopy(flatArray, 0, inputBuffer(i), batchIndex * inputBufferSize(i), inputBufferSize(i))
      }
      batchIndex += 1
    }
    setMinibatch(inputBuffer)
  }

  def forwardBackward(rowIt: Iterator[Row]): Unit = {
    print("entering forwardBackward\n")
    val t1 = System.currentTimeMillis()
    transformInto(rowIt, inputs)
    val t2 = System.currentTimeMillis()
    print("transformInto took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")
    forwardBackward()
  }

  def forward(rowIt: Iterator[Row], dataBlobNames: List[String] = List[String]()): Map[String, NDArray] = {
    transformInto(rowIt, inputs)
    forward(dataBlobNames)
  }

  def outputSchema(): StructType = {
    val fields = Array.range(0, numOutputs).map(i => {
      val output = caffeNet.blob_names().get(caffeNet.output_blob_indices().get(i)).getString
      new StructField(new String(output), DataTypes.createArrayType(DataTypes.FloatType), false)
    })
    StructType(fields)
  }

}