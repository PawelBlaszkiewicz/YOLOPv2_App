package com.example.yolopv2app

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import org.apache.commons.math3.analysis.function.Sigmoid
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.dnn.Dnn.*
import java.nio.FloatBuffer
import java.util.*
import kotlin.math.abs
import kotlin.math.absoluteValue
import kotlin.math.pow
import kotlin.math.sqrt

const val DIM_BATCH_SIZE = 1;
const val DIM_PIXEL_SIZE = 3;
const val IMAGE_SIZE_X = 320;
const val IMAGE_SIZE_Y = 192;

internal data class Result(
    var outputBitmap: Bitmap,
    var outputBox: Array<FloatArray>
)

internal class ObjectDetector{

    fun detect(
        imageBitmap: Bitmap,
        ortEnv: OrtEnvironment,
        ortSession: OrtSession,
        classes: List<String>
    ): Result {
        OpenCVLoader.initDebug()
        val mutableBitmap: Bitmap = imageBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val width = mutableBitmap.width.toFloat()
        val height = mutableBitmap.height.toFloat()
        val inputInfo = ortSession.inputInfo.values
        val inputShape = get_input_shape(inputInfo)
        val ratiow = (width / inputShape[3])
        val ratioh = (height / inputShape[2])

        val classes = classes;

        val rawBitmap =
            mutableBitmap.let { Bitmap.createScaledBitmap(it, inputShape[3], inputShape[2], false) }

        // Example values for old detection
        val row1 = floatArrayOf(
            500.061f,
            1500.0443f,
            430.07617f,
            670.5907f,
            0.8745549f,
            27.0f
        )
        val row2 = floatArrayOf(
            1000.6362f,
            701.31494f,
            350.27792f,
            710.1741f,
            0.70867646f,
            16.0f
        )

        val boxOutput: Array<FloatArray> = arrayOf(row1, row2)

        val imgData = preProcess(rawBitmap)
        val shape = longArrayOf(1, 3, 192, 320)
        val inputTensor = OnnxTensor.createTensor(ortEnv, imgData, shape)
        inputTensor.use {
            val output = ortSession.run(
                Collections.singletonMap("input", inputTensor),
                setOf("seg", "ll", "pred0", "pred1", "pred2")
            )
//####################################################################### OBJECT DETECTION CODE
            val confThreshold = 0.5F
            val pred = processPredictions(output)
            val det_out = processDetection(
                mutableBitmap,
                pred,
                confThreshold,
                ratiow,
                ratioh,
                classes,
                height.toInt()
            )
//####################################################################### ROAD_SEGMENTATION CODE
            val roadSegTest = output[0].value as Array<Array<Array<FloatArray>>>

            val drivable_area = extract3DArray(roadSegTest)
            val mask = argMax3D(drivable_area)
            val maskTransposed = transposeArray(mask)
            val scaledArray2 =
                scaleArray(maskTransposed, ratiow, ratioh, width.toInt(), height.toInt())
            val canvas2 = Canvas(det_out)
            val paint2 = Paint()
            paint2.color = Color.GREEN
            for (i in 0 until width.toInt()) {
                for (j in 0 until height.toInt()) {
                    val pixelValue = scaledArray2[i][j]
                    if (pixelValue == 1.0f) {
                        canvas2.drawPoint(i.toFloat(), j.toFloat(), paint2)
                    }
                }
            }
//####################################################################### LANE_LINE CODE
            val llTest = output[1].value as Array<Array<Array<FloatArray>>>

            val extractedArray = extract2DArray(llTest)

            modifyArray(extractedArray)

            val scaledArray = scaleArray(
                extractedArray,
                ratioh,
                ratiow,
                height.toInt(),
                width.toInt()
            )
            val pointDensity = 20
            val imgMiddle = Pair(width / 2, (0.85 * height).toInt())
            val bottomHorizon = Pair(imgMiddle.first.toInt(), (0.98 * height).toInt())
            val upperHorizon = Pair(imgMiddle.first.toInt(), (0.7 * height).toInt())
            val D = bottomHorizon.second - upperHorizon.second
            val threshold = sqrt((D / pointDensity * D / pointDensity).toDouble()) * 2.5

        // QUANTIFICATION
            val firstPhase = pointDensity / 3

            val leftLeftLanePoints = mutableListOf<Pair<Int, Int>>()
            val leftLanePoints = mutableListOf<Pair<Int, Int>>()
            val rightLanePoints = mutableListOf<Pair<Int, Int>>()
            val rightRightLanePoints = mutableListOf<Pair<Int, Int>>()
            var lastPoints = mutableListOf<Pair<Int, Int>>()


            for (i in 0 until pointDensity) {
                val lineHeight = bottomHorizon.second - (i * D / pointDensity)
                val points = findMiddlePixelOnHeight(scaledArray, lineHeight)
                if (i <= firstPhase) {
                    separatePoints(
                        points,
                        leftLeftLanePoints,
                        leftLanePoints,
                        rightLanePoints,
                        rightRightLanePoints,
                        imgMiddle.first.toInt()
                    )
                }
                else
                {
                    for (point in points) {
                        val howFar = 100000000000.0
                        findingClosestPointByWidthAndHeight(
                            point,
                            lastPoints,
                            howFar,
                            leftLanePoints,
                            rightLanePoints,
                            threshold
                        )
                    }
                }
                lastPoints = points
            }

        // DRAWING
            val canvas = Canvas(det_out)
            val paint = Paint()
            paint.color = Color.CYAN
            paint.strokeWidth = 10f
            for (point in leftLeftLanePoints) {
                canvas.drawPoint(point.first.toFloat(), point.second.toFloat(), paint)
            }
            paint.color = Color.rgb(255, 0, 0)
            for (point in leftLanePoints) {
                canvas.drawPoint(point.first.toFloat(), point.second.toFloat(), paint)
            }
            paint.color = Color.rgb(255, 150, 0)
            for (point in rightLanePoints) {
                canvas.drawPoint(point.first.toFloat(), point.second.toFloat(), paint)
            }
            paint.color = Color.rgb(180, 0, 255)
            for (point in rightRightLanePoints) {
                canvas.drawPoint(point.first.toFloat(), point.second.toFloat(), paint)
            }
            val result = Result(det_out, boxOutput)
            return result
        }
    }

    fun <T> List<T?>.firstNotNull(): T? {
        for (element in this) {
            if (element != null) {
                return element
            }
        }
        return null
    }
    fun averagePoint(listOfPoints: List<Pair<Int, Int>?>): Pair<Int, Int>? {
        var sum = 0
        var count = 0

        for (point in listOfPoints) {
            if (point != null) {
                sum += point.first
                count++
            }
        }
        if (count == 0) return null

        val averageValue = sum.toDouble() / count

        return Pair(averageValue.toInt(), listOfPoints.firstNotNull()?.second ?: return null)
    }

    fun findPointOnHeight(listOfPoints: List<Pair<Int, Int>>, height: Int): Pair<Int, Int>? {
        for (point in listOfPoints) {
            if (point.second == height) {
                return point
            }
        }
        return null
    }

    fun averageLine(setOfLines: List<List<Pair<Int, Int>>>): List<Pair<Int, Int>> {
        var b = 0
        var c = 0
        val workList = mutableListOf<List<Pair<Int, Int>>>()
        val finalList = mutableListOf<Pair<Int, Int>>()

        for (i in 0 until 5) {
            workList.add(setOfLines[setOfLines.size - i - 1])
        }

        for (i in 0 until 5) {
            val a = workList[i].size
            if (a > b) {
                b = a
                c = i
            }
        }
        val longestList = workList[c]
        workList.removeAt(c)

        for (point in longestList) {
            val allPointsOnHeight = mutableListOf(point)
            for (otherList in workList) {
                findPointOnHeight(otherList, point.second)?.let { allPointsOnHeight.add(it) }
            }
            averagePoint(allPointsOnHeight)?.let { finalList.add(it) }
        }
        return finalList
    }

    fun findingClosestPointByWidthAndHeight(
        point: Pair<Int, Int>,
        lastPoints: List<Pair<Int, Int>>,
        howFar: Double,
        rightLanePoints: MutableList<Pair<Int, Int>>,
        leftLanePoints: MutableList<Pair<Int, Int>>,
        threshold: Double
    ) {
        var howFarVar = howFar
        var index = 0
        for ((otherX, otherY) in lastPoints) {
            val howClose = sqrt(((point.first - otherX) * (point.first - otherX)).toDouble() + ((point.second - otherY) * (point.second - otherY)))
            if (howClose < howFarVar && howClose < threshold) {
                howFarVar = howClose
                index = lastPoints.indexOf(Pair(otherX, otherY))
            }
        }

        if (howFarVar != 100000000000.0) {
            checkPoints(rightLanePoints, point, lastPoints, index)
            checkPoints(leftLanePoints, point, lastPoints, index)
        }
    }

    fun checkPoints(
        lanePoints: MutableList<Pair<Int, Int>>,
        point: Pair<Int, Int>,
        lastPoints: List<Pair<Int, Int>>,
        index: Int
    ) {
        val threshold = 10
        if (lanePoints.size > 0 && point.second == lanePoints.last().second) {
            val a = sqrt(((point.first - lanePoints[lanePoints.size - 2].first) * (point.first - lanePoints[lanePoints.size - 2].first)).toDouble() + ((point.second - lanePoints[lanePoints.size - 2].second) * (point.second - lanePoints[lanePoints.size - 2].second)))
            val b = sqrt(((lanePoints.last().first - lanePoints[lanePoints.size - 2].first) * (lanePoints.last().first - lanePoints[lanePoints.size - 2].first)).toDouble() + ((lanePoints.last().second - lanePoints[lanePoints.size - 2].second) * (lanePoints.last().second - lanePoints[lanePoints.size - 2].second)))
            if (b > a) {
                lanePoints.removeAt(lanePoints.size - 1)
                lanePoints.add(point)
            }
        } else if (lanePoints.size > 0 && lastPoints[index] == lanePoints.last()) {
            lanePoints.add(point)
        } else if (lanePoints.size > 4 && lastPoints[index].second == lanePoints.last().second) {
            var a = 0
            var b = 0
            var c = 0
            var d = 0
            a = lanePoints[lanePoints.size - 3].first - lanePoints[lanePoints.size - 4].first
            b = lanePoints[lanePoints.size - 2].first - lanePoints[lanePoints.size - 3].first
            c = lanePoints.last().first - lanePoints[lanePoints.size - 2].first
            d = lastPoints[index].first - lanePoints[lanePoints.size - 2].first
            if (a > 0 && b > 0) {
                val diff1 = abs(c - (a + b) / 2)
                val diff2 = abs(d - (a + b) / 2)
                if (diff1 > diff2) {
                    lanePoints.removeAt(lanePoints.size - 1)
                    lanePoints.add(lastPoints[index])
                    lanePoints.add(point)
                }
            } else if (a < 0 && b < 0) {
                val diff1 = abs(c - (a + b) / 2)
                val diff2 = abs(d - (a + b) / 2)
                if (diff1 > diff2) {
                    lanePoints.removeAt(lanePoints.size - 1)
                    lanePoints.add(lastPoints[index])
                    lanePoints.add(point)
                }
            } else if (a > 0 && b < 0) {
                if (lastPoints[index].first < lanePoints[lanePoints.size - 2].first && abs(lastPoints[index].first - lanePoints[lanePoints.size - 2].first) < threshold) {
                    lanePoints.removeAt(lanePoints.size - 1)
                    lanePoints.add(lastPoints[index])
                    lanePoints.add(point)
                }
            } else if (a < 0 && b > 0) {
                if (lastPoints[index].first > lanePoints[lanePoints.size - 2].first && abs(lastPoints[index].first - lanePoints[lanePoints.size - 2].first) < threshold) {
                    lanePoints.removeAt(lanePoints.size - 1)
                    lanePoints.add(lastPoints[index])
                    lanePoints.add(point)
                }
            }
        }
    }

    fun separatePoints(
        points: List<Pair<Int, Int>>,
        leftLeftLanePoints: MutableList<Pair<Int, Int>>,
        leftLanePoints: MutableList<Pair<Int, Int>>,
        rightLanePoints: MutableList<Pair<Int, Int>>,
        rightRightLanePoints: MutableList<Pair<Int, Int>>,
        imgMiddle: Int
    ) {
        val leftDistanceList = mutableListOf<Int>()
        val rightDistanceList = mutableListOf<Int>()

        for (point in points) {
            val distance = point.first - imgMiddle
            if (distance < 0) {
                leftDistanceList.add(distance.absoluteValue)
            }
            if (distance >= 0) {
                rightDistanceList.add(distance.absoluteValue)
            }
        }

        if (leftDistanceList.isNotEmpty()) {
            val minLeftIdx = leftDistanceList.indexOf(leftDistanceList.minOrNull()!!)
            leftLanePoints.add(points[minLeftIdx])
            if (leftDistanceList.size > 1) {
                leftLeftLanePoints.add(points[minLeftIdx - 1])
            }
        }

        if (rightDistanceList.isNotEmpty()) {
            val minRightIdx = rightDistanceList.indexOf(rightDistanceList.minOrNull()!!)
            val adjustedMinRightIdx = minRightIdx + leftDistanceList.size
            rightLanePoints.add(points[adjustedMinRightIdx])
            if (rightDistanceList.size > 1) {
                rightRightLanePoints.add(points[adjustedMinRightIdx + 1])
            }
        }
    }

    fun findMiddlePixelOnHeight(laneMask: Array<FloatArray>, height: Int): MutableList<Pair<Int, Int>> {
        val horizontalLane = laneMask[height]
        var cnt0 = 0
        var cnt1 = 0
        var previousPixel = 0
        val pointsList = mutableListOf<Pair<Int, Int>>()

        for (currentPixel in horizontalLane.indices) {
            if ((horizontalLane[previousPixel] * horizontalLane[currentPixel]).toInt() == 1) {
                cnt1++
            } else if ((horizontalLane[previousPixel] * horizontalLane[currentPixel]).toInt() == 0 && cnt1 != 0) {
                pointsList.add(Pair(currentPixel - cnt1 / 2, height))
                cnt1 = 0
            } else if (horizontalLane[currentPixel].toInt() == 0) {
                cnt0++
            }
            previousPixel = currentPixel
        }

        return pointsList
    }

    fun argMax3D(array: Array<Array<FloatArray>>): Array<FloatArray> {
        require(array.isNotEmpty() && array[0].isNotEmpty() && array[0][0].isNotEmpty()) {
        }

        val height = array[0][0].size
        val width = array[0].size
        val result = Array(height) { FloatArray(width) }

        for (i in 0 until height) {
            for (j in 0 until width) {
                var maxIndex = 0
                var maxValue = array[0][j][i]

                for (k in 1 until array.size) {
                    if (array[k][j][i] > maxValue) {
                        maxIndex = k
                        maxValue = array[k][j][i]
                    }
                }

                result[i][j] = maxIndex.toFloat()
            }
        }

        return result
    }

    fun extract2DArray(array: Array<Array<Array<FloatArray>>>): Array<FloatArray>{
        val height_ll = array[0][0].size
        val width_ll = array[0][0][0].size

        val extractedArray = Array(height_ll) { FloatArray(width_ll) }

        for (i in 0 until height_ll) {
            for (j in 0 until width_ll) {
                extractedArray[i][j] = array[0][0][i][j]
            }
        }
        return extractedArray
    }

    fun extract3DArray(array: Array<Array<Array<FloatArray>>>): Array<Array<FloatArray>>{
        val height_ll = array[0][0].size
        val width_ll = array[0][0][0].size

        val extractedArray = Array(2) { Array(width_ll) { FloatArray(height_ll) } }

        for (i in 0 until height_ll) {
            for (j in 0 until width_ll) {
                extractedArray[0][j][i] = array[0][0][i][j]
                extractedArray[1][j][i] = array[0][1][i][j]
            }
        }
        return extractedArray
    }

    fun drawPred(bitmap: Bitmap, classId: Int, conf: Float, left: Int, top: Int, right: Int, bottom: Int, classes: List<String>, height: Int): Bitmap {
            val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(mutableBitmap)

            val paint = Paint().apply {
                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = (height/90).toFloat()
            }
            canvas.drawRect(left.toFloat(), top.toFloat(), right.toFloat(), bottom.toFloat(), paint)

            val label = classes[classId - 1] + ":" + String.format("%.2f", conf)

            val textPaint = Paint().apply {
                color = Color.GREEN
                textSize = (height/18).toFloat()
            }
            val textBounds: android.graphics.Rect = android.graphics.Rect()
            textPaint.getTextBounds(label, 0, label.length, textBounds)
            val textHeight = textBounds.height()

            canvas.drawText(label, left.toFloat(), maxOf(top - 10, textHeight).toFloat(), textPaint)

            return mutableBitmap
    }

    fun scaleArray(originalArray: Array<FloatArray>, ratiow: Float, ratioh: Float, width: Int, height: Int): Array<FloatArray> {
            val scaledArray = Array(width) { FloatArray(height) }

            for (x in 0 until width) {
                for (y in 0 until height) {
                    val originalX = (x * 1/ratiow).toInt()
                    val originalY = (y * 1/ratioh).toInt()

                    scaledArray[x][y] = originalArray[originalX][originalY]
                }
            }

            return scaledArray
        }

    fun transposeArray(array: Array<FloatArray>): Array<FloatArray> {
        val rows = array.size
        val cols = array[0].size

        val transposedArray = Array(cols) { FloatArray(rows) }

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                transposedArray[j][i] = array[i][j]
            }
        }
        return transposedArray
    }

    fun get_input_shape(inputInfo: MutableCollection<NodeInfo>): IntArray{
        for (node in inputInfo) {
            val infoString = node.info.toString()
            val shapeRegex = Regex("shape=\\[(.*?)]")
            val shapeMatch = shapeRegex.find(infoString)?.groupValues?.get(1)
            val shape: IntArray = shapeMatch?.split(",")?.map { it.trim().toInt() }?.toIntArray()!!
            return shape
        }
        return intArrayOf(0)
    }

    fun processDetection(bitmap: Bitmap, pred: ArrayList<FloatArray>, confThreshold: Float, ratiow: Float, ratioh: Float, classes: List<String>, height_txt: Int): Bitmap {
        val boxes = mutableListOf<Rect2d>()
        val confidences = mutableListOf<Float>()
        val classIds = mutableListOf<Int>()
        var processedBitmap = bitmap

        for (i in pred.indices) {
            val score = pred[i][4] * pred[i].sliceArray(5 until pred[i].size).max()

            if (score < confThreshold) {
                continue
            }

            val maxInArray = pred[i].sliceArray(5 until pred[i].size).max() // tu 0 zamiast 5 i będzie git???
            val classId = pred[i].indexOfFirst { it == maxInArray }
            val (cx, cy, w, h) = pred[i].sliceArray(0 until 4)
            val x = ((cx - 0.5 * w) * ratiow).toInt()
            val y = ((cy - 0.5 * h) * ratioh).toInt()
            val width = (w * ratiow).toInt()
            val height = (h * ratioh).toInt()

            boxes.add(Rect2d(x.toDouble(), y.toDouble(), width.toDouble(), height.toDouble()))
            classIds.add(classId-5)
            confidences.add(score)
        }
        val matOfRect2d = MatOfRect2d()
        matOfRect2d.fromList(boxes)

        val matOfFloat = MatOfFloat()
        matOfFloat.fromList(confidences)

        val indices = MatOfInt()
        NMSBoxes(matOfRect2d, matOfFloat, confThreshold, 0.5F, indices)
        val indicesArray = IntArray(indices.rows())
        indices.get(0, 0, indicesArray)

        for (i in 0 until indices.rows()) {
            val box = boxes[indices[i, 0][0].toInt()]
            val left = box.x.toInt()
            val top = box.y.toInt()
            val width = box.width.toInt()
            val height = box.height.toInt()


            processedBitmap = drawPred(processedBitmap, classIds[i], confidences[i], left, top, left + width, top + height, classes, height_txt)
        }
        return processedBitmap
    }
    fun processPredictions(results: OrtSession.Result): ArrayList<FloatArray> {
        val predArray = ArrayList<FloatArray>()
        val stride = listOf(8, 16, 32)
        val anchors = listOf(
            listOf(12f, 16f, 19f, 36f, 40f, 28f),
            listOf(36f, 75f, 76f, 55f, 72f, 146f),
            listOf(142f, 110f, 192f, 243f, 459f, 401f)
        )
        val grid = ArrayList<Array<Array<FloatArray>>>()
        generateGrid(grid, 192, 320, stride)
        val anchorsArray = anchors.map { it.toFloatArray() }.toTypedArray()
        val anchorsTensor = Array(3) { Array(3) { Array(1) { Array(1) { FloatArray(2) } } } }

        for (i in 0 until 3) {
            for (j in 0 until 3) {
                for (k in 0 until 2) {
                    anchorsTensor[i][j][0][0][k] = anchorsArray[i][j*2+k]
                }
            }
        }

        for (i in 2 until 5) {
            val j = i-2
            val layerResults = results[i].value as Array<Array<Array<FloatArray>>>
            val bs = layerResults.size
            val ny = layerResults[0][0].size
            val nx = layerResults[0][0][0].size

            val y = Array(bs) { Array(3) { Array(85) { Array(ny) { FloatArray(nx) } } } }

            for (a in 0 until bs) {
                for (b in 0 until 3) {
                    for (c in 0 until 85) {
                        for (yy in 0 until ny) {
                            for (xx in 0 until nx) {
                                y[a][b][c][yy][xx] = layerResults[a][b * 85 + c][yy][xx]
                            }
                        }
                    }
                }
            }

            val transposedArray = Array(bs) { Array(3) { Array(ny) { Array(nx) { FloatArray(85) } } } }

            for (a in 0 until bs) {
                for (b in 0 until 3) {
                    for (yy in 0 until ny) {
                        for (xx in 0 until nx) {
                            for (c in 0 until 85) {
                                transposedArray[a][b][yy][xx][c] = y[a][b][c][yy][xx]
                            }
                        }
                    }
                }
            }

            sigmoidActivation(transposedArray)

            for (a in 0 until bs) {
                for (b in 0 until 3) {
                    for (yy in 0 until ny) {
                        for (xx in 0 until nx) {
                            transposedArray[a][b][yy][xx][0] =
                                ((transposedArray[a][b][yy][xx][0] * 2.0 - 0.5 + grid[j][yy][xx][0]) * stride[j]).toFloat() //stride[0]=8
                            transposedArray[a][b][yy][xx][1] =
                                ((transposedArray[a][b][yy][xx][1] * 2.0 - 0.5 + grid[j][yy][xx][1]) * stride[j]).toFloat() //stride[0]=8
                            transposedArray[a][b][yy][xx][2] =
                                ((transposedArray[a][b][yy][xx][2] * 2).pow(2) * anchorsTensor[j][b][0][0][0])
                            transposedArray[a][b][yy][xx][3] =
                                ((transposedArray[a][b][yy][xx][3] * 2).pow(2) * anchorsTensor[j][b][0][0][1])
                        }
                    }
                }
            }
            val reshapedArray = Array(3 * ny * nx) { FloatArray(85) }
            var index = 0
            for (a in 0 until bs) {
                for (b in 0 until 3) {
                    for (yy in 0 until ny) {
                        for (xx in 0 until nx) {
                            for (c in 0 until 85) {
                                reshapedArray[index][c] = transposedArray[a][b][yy][xx][c]
                            }
                            predArray.add(reshapedArray[index])
                            index += 1
                        }
                    }
                }
            }
        }
        return predArray
    }

    private fun sigmoidActivation(y: Array<Array<Array<Array<FloatArray>>>>) {
        for (a in y.indices) {
            for (b in y[0].indices){
                for (yy in y[a][b].indices) {
                    for (xx in y[a][b][yy].indices) {
                        for (c in y[a][b][yy][xx].indices) {
                            y[a][b][yy][xx][c] = Sigmoid().value(y[a][b][yy][xx][c].toDouble()).toFloat()
                        }
                    }
                }
            }
        }
    }

    private fun generateGrid(grid: ArrayList<Array<Array<FloatArray>>>,inputHeight: Int, inputWidth: Int, stride: List<Int>) {
        for (i in 0 until 3) {
            val h = inputHeight / stride[i]
            val w = inputWidth / stride[i]
            grid.add(makeGrid(w, h))
        }
    }

    private fun makeGrid(nx: Int, ny: Int): Array<Array<FloatArray>> {
        val xv = Array(nx) { it.toFloat() }
        val yv = Array(ny) { it.toFloat() }

        val gridArray = Array(ny) { Array(nx) { FloatArray(2) } }

        for (yy in 0 until ny) {
            for (xx in 0 until nx) {
                gridArray[yy][xx][0] = xv[xx]
                gridArray[yy][xx][1] = yv[yy]
            }
        }

        return gridArray
    }

    fun modifyArray(array: Array<FloatArray>) {
            val rows = array.size
            val cols = array[0].size

            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    val originalValue = array[i][j]

                    if (originalValue < 0.5f) {
                        array[i][j] = 0.0f
                    } else if (originalValue <= 1f) {
                        array[i][j] = 1.0f
                    }
                }
            }
        }

    fun preProcess(bitmap: Bitmap): FloatBuffer {
        val imgData = FloatBuffer.allocate(
            DIM_BATCH_SIZE
                    * DIM_PIXEL_SIZE
                    * IMAGE_SIZE_X
                    * IMAGE_SIZE_Y
        )
        imgData.rewind()
        val stride = IMAGE_SIZE_X * IMAGE_SIZE_Y
        val bmpData = IntArray(stride)

        bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (i in 0..IMAGE_SIZE_X - 1) {
            for (j in 0..IMAGE_SIZE_Y - 1) {
                val idx = IMAGE_SIZE_Y * i + j
                val pixelValue = bmpData[idx]
                imgData.put(idx,(pixelValue shr 16 and 0xFF) / 255f)
                imgData.put(idx + stride, (pixelValue shr 8 and 0xFF) / 255f)
                imgData.put(idx + stride * 2, (pixelValue and 0xFF) / 255f)
            }
        }

        imgData.rewind()
        return imgData
    }

}