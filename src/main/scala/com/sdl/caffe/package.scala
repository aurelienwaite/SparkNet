package com.sdl

/**
  * Created by rorywaite on 23/03/2016.
  */
package object caffe{

  case class Blob(shape: Array[Int], data: Array[Float]) extends Ordered[Blob]{
    override def toString() = {
      "Shape: " + shape.mkString(",") + "; Data: " + data.mkString(",")
    }

    override def compare(that: Blob) =
      Ordering[(Iterable[Int], Iterable[Float])].compare((this.shape, this.data), (that.shape, that.data))

  }

  type Weights = Map[String, scala.collection.immutable.IndexedSeq[Blob]]

  //Scala has a meltdown trying to find the implicit for the weight type. This helps it along
  val blobOrdering = Ordering.Iterable[Blob]
  val layerOrdering = Ordering.Tuple2(Ordering[String], blobOrdering)
  val weightOrdering = Ordering.Iterable[(String, Iterable[Blob])](layerOrdering)

}
