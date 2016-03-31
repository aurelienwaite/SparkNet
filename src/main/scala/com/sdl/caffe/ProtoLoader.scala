// Utilities for loading models from caffe .prototxt files

package com.sdl.caffe

import caffe._
import caffe.Caffe._

object ProtoLoader {
  def loadSolverPrototxt(filename: String, libLocation: String) : SolverParameter = {
    val caffeLib = CaffeLibrary.Instance.get(libLocation)
    val state = caffeLib.create_state()
    caffeLib.parse_solver_prototxt(state, filename)
    var len = caffeLib.get_prototxt_len(state)
    var data = caffeLib.get_prototxt_data(state)
    var bytes = data.getByteArray(0, len)
    caffeLib.destroy_state(state)
    return SolverParameter.parseFrom(bytes)
  }

  def loadNetPrototxt(filename: String, libLocation: String): NetParameter = {
    val caffeLib = CaffeLibrary.Instance.get(libLocation)
    val state = caffeLib.create_state()
    caffeLib.parse_net_prototxt(state, filename)
    var len = caffeLib.get_prototxt_len(state)
    var data = caffeLib.get_prototxt_data(state)
    var bytes = data.getByteArray(0, len)
    caffeLib.destroy_state(state)
    return NetParameter.parseFrom(bytes)
  }


  def replaceDataLayers(netParameter: NetParameter, trainBatchSize: Int, testBatchSize: Int, numChannels: Int, height: Int, width: Int): NetParameter = {
    val netBuilder = netParameter.toBuilder()
    netBuilder.setLayer(0, RDDLayer("data", shape=List(trainBatchSize, numChannels, height, width), Some(Include.Train)))
    netBuilder.setLayer(1, RDDLayer("label", shape=List(trainBatchSize, 1), Some(Include.Train)))
    netBuilder.addLayer(0, RDDLayer("data", shape=List(testBatchSize, numChannels, height, width), Some(Include.Test)))
    netBuilder.addLayer(1, RDDLayer("label", shape=List(testBatchSize, 1), Some(Include.Test)))
    return netBuilder.build()
  }
}
