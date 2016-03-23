package libs

import scala.collection.mutable

/**
  * Created by rorywaite on 23/03/2016.
  */


case class RichWeights(w: Weights){

  private def zip(other: Weights) = {
    def msg = s"Layers do not match: ${w.keys} vs ${other.keys}"
    assert(w.keys.size == other.keys.size, msg)
    val keys = w.keys ++ other.keys
    assert(keys.size == w.keys.size, msg)
    for(k <- keys) yield {
      val blobs = w(k)
      val otherBlobs = other(k)
      assert(blobs.size == otherBlobs.size, s"For layer $k there are a different number of blobs ${blobs.size} vs ${otherBlobs.size}")
      val zipped = for((b, otherB) <- blobs zip otherBlobs) yield {
        assert(b.shape.sameElements(otherB.shape),
          s"For layer $k blobs shape do match ${b.shape.deep} vs ${otherB.shape.deep}")
        (b.shape, b.data, otherB.data)
      }
      k -> zipped
    }
  }

  private def applyOp(op: (Float, Float) => Float, other: Weights): Weights = {
    val res = for {
      (k, zipped) <- zip(other)
    } yield
      k -> (for ((shape, data, otherData) <- zipped) yield
        {
          val applied = for ((f1, f2) <- data zip otherData) yield op(f1, f2)
          Blob(shape, applied)
        })
    res.toMap
  }

  def add(other: Weights) = applyOp((_+_), other)

  def subtract(other: Weights) = applyOp((_-_), other)

  def scalarDivide(v: Float): Weights =
    for ((k, blobs) <- w) yield
      k -> (for (b <- blobs) yield
        Blob(b.shape, b.data.map(_ / v)))


  def /(v: Float) = scalarDivide(v)


}


object RichWeights{

  implicit def weightsToRichWeights(w: Weights) = RichWeights(w)

}
