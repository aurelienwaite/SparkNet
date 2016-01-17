package libs

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import scala.collection.mutable.Map
import scala.collection.mutable.MutableList

trait NetInterface {
  def forward(rowIt: Iterator[Row]): Array[Row]
  def forwardBackward(rowIt: Iterator[Row])
  def getWeights(): WeightCollection
  def setWeights(weights: WeightCollection)
  def outputSchema(): StructType
}

class JavaCPPCaffeNet(netParam: NetParameter, schema: StructType, preprocessor: Preprocessor) {
  private val inputSize = netParam.input_size
  private val batchSize = netParam.input_shape(0).dim(0).toInt
  private val transformations = new Array[Any => NDArray](inputSize)
  private val inputIndices = new Array[Int](inputSize)
  private val columnNames = schema.map(entry => entry.name)
  private val caffeNet = new FloatNet(netParam)
  private val numOutputs = caffeNet.num_outputs

  for (i <- 0 to inputSize - 1) {
    val name = netParam.input(i).getString
    transformations(i) = preprocessor.convert(name, getInputShape(i))
    inputIndices(i) = columnNames.indexOf(name)
  }

  // Preallocate a buffer for data input into the net
  val inputs = new FloatBlobVector(inputSize)
  for (i <- 0 to inputSize - 1) {
    val dims = new Array[Int](netParam.input_shape(i).dim_size)
    for (j <- dims.indices) {
      dims(j) = netParam.input_shape(i).dim(j).toInt
    }
    inputs.put(i, new FloatBlob(dims))
  }

  def transformInto(iterator: Iterator[Row], data: FloatBlobVector): Unit = {
    var batchIndex = 0
    while (iterator.hasNext) {
      val row = iterator.next
      for (i <- 0 to inputSize - 1) {
        val result = transformations(i)(row(inputIndices(i)))
        val flatArray = result.toFlat() // TODO: Make this efficient
        val blob = data.get(i)
        val buffer = blob.cpu_data()
        val offset = blob.offset(batchIndex)
        buffer.position(offset)
        buffer.put(flatArray: _*)
      }
      batchIndex += 1
      if (batchIndex == batchSize) {
        return
      }
    }
  }

  def forward(rowIt: Iterator[Row]): Array[Row] = {
    val result = new Array[Array[Float]](batchSize)
    transformInto(rowIt, inputs)
    val tops = caffeNet.Forward(inputs)
    for (i <- 0 to batchSize - 1) {
      val top = tops.get(0)
      val shape = Array.range(0, top.num_axes).map(i => top.shape.get(i))
      val array = new Array[Float](shape.product)
      val data = top.cpu_data()
      data.get(array)
      print("len of array ", array.length)
      result(i) = array
    }
    return result.map(row => Row(row))
  }

  def forwardBackward(rowIt: Iterator[Row]) = {
    transformInto(rowIt, inputs)
    caffeNet.ForwardBackward(inputs)
  }

  def getWeights(): WeightCollection = {

    return new WeightCollection(Map[String, MutableList[NDArray]](), List[String]())
  }

  def setWeights(weights: WeightCollection) = {
    // TODO: check that weights.keys agrees with netParam, or something like that
    for (i <- 0 to caffeNet.layers().size.toInt) {
      for (j <- 0 to caffeNet.layers().get(i).blobs().size.toInt) {
        val blob = caffeNet.layers().get(i).blobs().get(j)
        val shape = getShape(blob)
      }
    }

    // TODO: incomplete

    /*
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
    */
  }

  private def getShape(blob: FloatBlob): Array[Int] = {
    val numAxes = blob.num_axes()
    val shape = new Array[Int](numAxes)
    for (k <- 0 to numAxes - 1) {
      shape(k) = blob.shape(k)
    }
    return shape
  }

  private def getInputShape(i: Int): Array[Int] = {
    val numAxes = netParam.input_shape(i).dim_size - 1
    val shape = new Array[Int](numAxes)
    for (j <- 0 to numAxes - 1) {
      shape(j) = netParam.input_shape(i).dim(j + 1).toInt
    }
    return shape
  }

  def outputSchema(): StructType = {
    val fields = Array.range(0, numOutputs).map(i => {
      val output = caffeNet.blob_names().get(caffeNet.output_blob_indices().get(i)).getString
      new StructField(output, DataTypes.createArrayType(DataTypes.FloatType), false)
    })
    StructType(fields)
  }
}
