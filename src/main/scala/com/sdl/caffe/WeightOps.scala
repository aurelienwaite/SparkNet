package com.sdl.caffe

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, DataInputStream, DataOutputStream}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.compress._
import org.apache.hadoop.io.compress.zlib.ZlibCompressor.CompressionLevel

import scala.language.implicitConversions

/**
  * Utility function for dealing with weights serialized to a byte array. There is some complication to how compression
  * is handled.
  *
  * - Model weights should never be compressed because model weights are close to being randomly distributed
  * - Model diffs should always be compressed because they are sparse
  *
  * The operations ensure that we never have to decompress diffs
  */


object WeightOps{

  private type CodecOpt = Option[CompressionCodec]

  private val codec = new DeflateCodec()
  private val conf = new Configuration()
  conf.setEnum("zlib.compress.level", CompressionLevel.BEST_SPEED)
  codec.setConf(conf) // If not set it will generate a null pointer exception
  private val codecOpt = Option(codec)

  private def applyFunc[S, T](thisIn: S, thatIn: S, op: S => T): (T, T) = (op(thisIn), op(thatIn))

  private def makeDataIn(bytes: Array[Byte], compress: CodecOpt) = {
    val bytesIn = new ByteArrayInputStream(bytes)
    val opt = compress.map(_.createInputStream(bytesIn)).getOrElse(bytesIn)
    new DataInputStream(opt)
  }

  private def makeDataOut(compress: CodecOpt) = {
    val bytesOut = new ByteArrayOutputStream()
    val opt = compress.map(_.createOutputStream(bytesOut)).getOrElse(bytesOut)
    val out = new DataOutputStream(opt)
    (bytesOut, out)
  }

  private def readShape(in: DataInputStream) = {
    val length = in.readInt()
    for(_ <- 0 until length) yield in.readInt()
  }

  private def writeShape(toWrite: IndexedSeq[Int], out: DataOutputStream) = {
    out.writeInt(toWrite.length)
    for(s <- toWrite) out.writeInt(s)
  }

  private def applyOp(op: (Float, Float) => Float)
                     (thisCompress: CodecOpt, thatCompress: CodecOpt, outCompress: CodecOpt)
                     (thisW: Array[Byte], thatW: Array[Byte]): Array[Byte]= {
    val (bytesOut, out) = makeDataOut(outCompress)
    val (thisIn, thatIn) = (makeDataIn(thisW, thisCompress), makeDataIn(thatW, thatCompress))
    def applyToIn[T](op: DataInputStream => T) = applyFunc(thisIn, thatIn, op)
    def readInts = applyToIn(_.readInt())
    val (thisNumLayers, thatNumLayers) = readInts
    assert(thisNumLayers == thatNumLayers, s"Layers do not match: $thisNumLayers vs $thatNumLayers")
    out.writeInt(thisNumLayers)
    for(l <- 0 until thisNumLayers) {
      val (thisLayerName, thatLayerName) = applyToIn(_.readUTF())
      assert(thisLayerName == thatLayerName, s"The names do not match for the ${l}th layer, $thisLayerName vs $thatLayerName")
      out.writeUTF(thisLayerName)
      val (thisNumBlobs, thatNumBlobs) = readInts
      assert(thisNumBlobs == thatNumBlobs, s"For layer $thisLayerName there are a different number of blobs $thisNumBlobs vs $thatNumBlobs")
      out.writeInt(thisNumBlobs)
      for(b <- 0 until thisNumBlobs) {
        val (thisShape, thatShape) = applyToIn(readShape)
        assert(thisShape == thatShape, s"Blob $b shapes do not match $thisShape vs $thatShape")
        writeShape(thisShape, out)
        for(_ <-0 until thisShape.product){
          val(thisF, thatF) = applyToIn(_.readFloat())
          out.writeFloat(op(thisF, thatF))
        }
      }
    }
    applyToIn(_.close())
    out.close()
    bytesOut.toByteArray
  }

  /**
    * For adding weights back into the net. The first argument is uncompressed, and the second argument is compressed.
    * The output is compressed
    */
  val netAdd =
    applyOp(_+_)(thisCompress = None, thatCompress = codecOpt, outCompress = None)_

  /**
    * For use during reduces. All input and output is compressed
    */
  val diffAdd =
    applyOp(_+_)(thisCompress = codecOpt, thatCompress =  codecOpt, outCompress =  codecOpt)_

  /**
    * Subtract model weights to get compressed diffs
    */
  val subtract =
    applyOp(_-_)(thisCompress = None, thatCompress = None, outCompress = codecOpt)_

  /**
    * In place divide for diffs. Input and output compressed
    *
    * @param diff Compressed diff
    * @param v divisor
    */
  def scalarDivide(diff: Array[Byte], v: Float) = {
    val (bytesOut, out) = makeDataOut(codecOpt)
    val in = makeDataIn(diff, codecOpt)
    val numLayers = in.readInt()
    out.writeInt(numLayers)
    for(_ <- 0 until numLayers) {
      val layerName = in.readUTF()
      out.writeUTF(layerName)
      val numBlobs = in.readInt()
      out.writeInt(numBlobs)
      for(_ <- 0 until numBlobs) {
        val shape = readShape(in)
        writeShape(shape, out)
        for (_ <- 0 until shape.product) out.writeFloat(in.readFloat() / v)
      }
    }
    in.close()
    out.close()
    bytesOut.toByteArray
  }


}

