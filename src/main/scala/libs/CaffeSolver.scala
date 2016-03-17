package libs

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

trait Solver {
  def step(rowIt: Iterator[Row])
}

class CaffeSolver(solverParam: SolverParameter, schema: StructType, preprocessor: Preprocessor) extends FloatSGDSolver(solverParam) {

  val trainNet = new DFCaffeNet(solverParam.net_param, net(), schema, preprocessor)

  def step(rowIt: Iterator[Row]) {
    trainNet.forwardBackward(rowIt)
    super.ApplyUpdate()
  }

}
