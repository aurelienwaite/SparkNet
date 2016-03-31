// This file should be extended and tested more

package com.sdl.caffe

import caffe._
import caffe.Caffe._
import scala.collection.JavaConverters._

object Include extends Enumeration {
  type Include = Value
  val Train, Test = Value
}

import com.sdl.caffe.Include._

object RDDLayer {
  def apply(name: String, shape: List[java.lang.Long], include: Option[Include]) : LayerParameter = {
    val blobShape = BlobShape.newBuilder()
    blobShape.addAllDim(shape.asJava)
    val layerParam = JavaDataParameter.newBuilder()
    layerParam.setShape(blobShape)
    val result = LayerParameter.newBuilder()
    result.setType("JavaData")
    result.setName(name)
    result.addAllTop(List(name).asJava)
    result.setJavaDataParam(layerParam)
    if (include != None) {
      val netState = NetStateRule.newBuilder()
      if (include.get == Include.Train) {
        netState.setPhase(Phase.TRAIN)
      } else {
        netState.setPhase(Phase.TEST)
      }
      result.addInclude(netState)
    }
    return result.build()
  }


}
