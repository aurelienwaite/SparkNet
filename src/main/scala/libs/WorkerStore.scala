package libs

import libs._

class WorkerStore() {
  var nets: Map[String, CaffeNet] = Map()
  var caffeLib: Option[CaffeLibrary] = None

  def setNet(name: String, net: CaffeNet) = {
    nets += (name -> net)
    setLib(net.caffeLib)
  }

  def getNet(name: String): CaffeNet = {
    return nets(name)
  }

  def initialized() = caffeLib.isDefined

  def setLib(library: CaffeLibrary) = {
    caffeLib = Option(library)
  }

  def getLib(): CaffeLibrary = {
    assert(!caffeLib.isEmpty)
    return caffeLib.get
  }

  def reset() = {
    nets = Map()
    for(_ <- caffeLib) {
      CaffeLibrary.Instance.dispose()
    }
    caffeLib = None
  }

}
