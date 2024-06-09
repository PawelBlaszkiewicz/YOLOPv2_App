package com.example.yolopv2app

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Typeface
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.apache.commons.math3.analysis.function.Sigmoid
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction
import org.apache.commons.math3.fitting.PolynomialCurveFitter
import org.apache.commons.math3.fitting.WeightedObservedPoints
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfRect2d
import org.opencv.core.Rect2d
import org.opencv.dnn.Dnn.NMSBoxes
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.collections.ArrayList
import kotlin.collections.List
import kotlin.collections.MutableCollection
import kotlin.collections.MutableList
import kotlin.collections.component1
import kotlin.collections.component2
import kotlin.collections.component3
import kotlin.collections.component4
import kotlin.collections.contentToString
import kotlin.collections.find
import kotlin.collections.first
import kotlin.collections.indexOfFirst
import kotlin.collections.indices
import kotlin.collections.isNotEmpty
import kotlin.collections.last
import kotlin.collections.listOf
import kotlin.collections.map
import kotlin.collections.max
import kotlin.collections.minOrNull
import kotlin.collections.mutableListOf
import kotlin.collections.setOf
import kotlin.collections.sliceArray
import kotlin.collections.toDoubleArray
import kotlin.collections.toFloatArray
import kotlin.collections.toIntArray
import kotlin.collections.toList
import kotlin.collections.toMutableList
import kotlin.collections.toTypedArray
import kotlin.math.abs
import kotlin.math.absoluteValue
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.reflect.KFunction1

internal class CameraYolopv2Detector(
    private val callBack: KFunction1<Pair<Bitmap, Int>, Unit>,
    private val ortEnv: OrtEnvironment,
    private val ortSession: OrtSession,
    private val classes: List<String>,
    private val setOfLanesLeft: MutableList<MutableList<Pair<Int, Int>>>,
    private val setOfLanesRight: MutableList<MutableList<Pair<Int, Int>>>,
    private val safetyFlag: Int,
    private var carFlag: Int)
    : ImageAnalysis.Analyzer {

    override fun analyze(
        image: ImageProxy,
    ){
        val mutableBitmap1: Bitmap = image.toBitmap()
        val mutableBitmap = mutableBitmap1.rotate(image.imageInfo.rotationDegrees.toFloat())
        val width = mutableBitmap.width.toFloat()
        val height = mutableBitmap.height.toFloat()
        val inputInfo = ortSession.inputInfo.values
        val inputShape = get_input_shape(inputInfo)
        val ratiow = (width / inputShape[3])
        val ratioh = (height / inputShape[2])
        var flag = safetyFlag
        var cFlag = carFlag

        val classes = classes;
        var processTimeMs: Long = 0
        var processingTimeMs: Long = 0

        val rawBitmap =
            mutableBitmap.let { Bitmap.createScaledBitmap(it, inputShape[3], inputShape[2], false) }

        val imgData = preProcess(rawBitmap)
        val shape = longArrayOf(1, 3, 192, 320)
        val inputTensor = OnnxTensor.createTensor(ortEnv, imgData, shape)
        var startTime = SystemClock.uptimeMillis()
        inputTensor.use {
            val output = ortSession.run(
                Collections.singletonMap("input", inputTensor),
                setOf("seg", "ll", "pred0", "pred1", "pred2")
            )
            processTimeMs = SystemClock.uptimeMillis() - startTime
            startTime = SystemClock.uptimeMillis()
//####################################################################### OBJECT DETECTION CODE
            val confThreshold = 0.5F
            val pred = processPredictions(output)
            val detectionOut = processDetection(
                mutableBitmap,
                pred,
                confThreshold,
                ratiow,
                ratioh,
                classes,
                height.toInt()
            )
            var det_out = detectionOut.first
            val carList = detectionOut.second
//####################################################################### ROAD_SEGMENTATION CODE
            val roadSegTest = output[0].value as Array<Array<Array<FloatArray>>>

            val drivable_area = extract3DArray(roadSegTest)
            val mask = argMax3D(drivable_area)
            val maskTransposed = transposeArray(mask)
            val scaledArray2 =
                scaleArray(maskTransposed, ratiow, ratioh, width.toInt(), height.toInt())
            val draw = 0
            var topPointDriveableArea = findHighestPixel(scaledArray2)
            if (draw == 1) {
                val canvas2 = Canvas(det_out)
                val paint2 = Paint()
                paint2.color = Color.GREEN
                for (i in 0 until width.toInt()) {
                    for (j in 0 until height.toInt()) {
                        val pixelValue = scaledArray2[i][j]
                        if (pixelValue == 1.0f) {
                            // If the value is 1, draw a green pixel at (j, i) on the canvas
                            canvas2.drawPoint(i.toFloat(), j.toFloat(), paint2)
                        }
                    }
                }
            }


//####################################################################### LANE_LINE CODE
            val llTest = output[1].value as Array<Array<Array<FloatArray>>>

            val extractedArray = extract2DArray(llTest)

            modifyArray(extractedArray) // 0-1

            val scaledArray = scaleArray(
                extractedArray,
                ratioh,
                ratiow,
                height.toInt(),
                width.toInt()
            )
            val pointDensity = 13
            val imgMiddle = Pair(width / 2, (0.85 * height).toInt())
            val bottomHorizon = Pair(imgMiddle.first.toInt(), (0.98 * height).toInt())
            val upperHorizon = Pair(imgMiddle.first.toInt(), topPointDriveableArea.second)
            val D = bottomHorizon.second - upperHorizon.second
            val threshold = sqrt((D / pointDensity * D / pointDensity).toDouble()) * 2.5
        // QUANTIFICATION
            val firstPhase = pointDensity / 2

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
            clearIfTooMany(setOfLanesLeft, 4)
            clearIfTooMany(setOfLanesRight, 4)
            appendingListIfFoundOrNot(leftLanePoints, setOfLanesLeft)
            appendingListIfFoundOrNot(rightLanePoints, setOfLanesRight)

            var leftPolyDegree = 1
            var rightPolyDegree = 1
            if (leftLanePoints.size > 10) {
                leftPolyDegree = 2
            }
            if (rightLanePoints.size > 10) {
                rightPolyDegree = 2
            }

            val leftLine = approximateLine(setOfLanesLeft, leftPolyDegree, upperHorizon, bottomHorizon, "left")
            val rightLine = approximateLine(setOfLanesRight, rightPolyDegree, upperHorizon, bottomHorizon, "right")
            val correctedRightPoint = rightLine.find { it.second.toFloat().toInt() == topPointDriveableArea.second}
            val correctedLeftPoint = leftLine.find { it.second.toFloat().toInt() == topPointDriveableArea.second}
            val correctedPoints = Pair(correctedLeftPoint, correctedRightPoint)

        // Top road horizon clarification with lines
            if (leftLine.isNotEmpty() and rightLine.isNotEmpty()) {
                topPointDriveableArea =
                    findHighestPixelClarified(scaledArray2, leftLine.first(), rightLine.first())
            }

            val carMiddle = Pair(width/2, (topPointDriveableArea.second+((height - topPointDriveableArea.second)/2)))
            val out : Triple<Bitmap, Int, Int>
            if (leftLine.isNotEmpty() and rightLine.isNotEmpty()) {
                out = processWarning(det_out, leftLine, rightLine, carMiddle, flag, carList, correctedPoints)
                det_out = out.first
                flag = out.second
                cFlag = out.third
            }

        // DRAWING
            val canvas = Canvas(det_out)
            val paint = Paint()
            paint.strokeWidth = 10f

            paint.color = Color.rgb(255, 128, 0) // carMiddle point for warning
            canvas.drawPoint(carMiddle.first, carMiddle.second, paint)


            paint.strokeWidth = 3f
            paint.color = Color.rgb(255, 255, 0)
            if (rightLine.isNotEmpty()) {
                for (point in rightLine) {
                    if (point.second >= topPointDriveableArea.second) {
                        canvas.drawPoint(point.first.toFloat(), point.second.toFloat(), paint)
                        if (carMiddle.second.toInt() == point.second) {
                            paint.color = Color.rgb(255, 128, 0)
                            paint.strokeWidth = 10f
                            canvas.drawPoint(point.first.toFloat(), carMiddle.second, paint)
                            paint.strokeWidth = 3f
                            paint.color = Color.rgb(255, 255, 0)
                        }
                    }
                }
            }
            if (leftLine.isNotEmpty()){
                for (point in leftLine) {
                    if(point.second >= topPointDriveableArea.second){
                        canvas.drawPoint(point.first.toFloat(), point.second.toFloat(), paint)
                        if (carMiddle.second.toInt() == point.second){
                            paint.color = Color.rgb(255, 128, 0)
                            paint.strokeWidth = 10f
                            canvas.drawPoint(point.first.toFloat(), carMiddle.second, paint)
                            paint.strokeWidth = 3f
                            paint.color = Color.rgb(255, 255, 0)
                        }
                    }
                }
            }
//################################# TOP LINE ##############################
            paint.color = Color.YELLOW
            paint.strokeWidth = 2f
            if (leftLine.isNotEmpty() and rightLine.isNotEmpty()){
                val rPoint = rightLine.find { it.second.toFloat().toInt() == topPointDriveableArea.second}
                val lPoint = leftLine.find { it.second.toFloat().toInt() == topPointDriveableArea.second}
                if (rPoint != null && lPoint != null) {
                    canvas.drawLine(
                        lPoint.first.toFloat(),
                        topPointDriveableArea.second.toFloat(),
                        rPoint.first.toFloat(),
                        topPointDriveableArea.second.toFloat(),
                        paint
                    )
                }
            }

            processingTimeMs = SystemClock.uptimeMillis() - startTime
            val rounded1 = String.format("%.2f", processTimeMs / 1000f)
            val rounded2 = String.format("%.2f", processingTimeMs / 1000f)
            drawFPS(canvas, det_out.height, det_out.width, "Inf: ${rounded1}s, Proc: ${rounded2}s", Color.BLACK)
            callBack(Pair(det_out, cFlag))
        }
    image.close()
    }
    fun processWarning(
        bitmap: Bitmap,
        leftLane: List<Pair<Int, Int>>,
        rightLane: List<Pair<Int, Int>>,
        carMiddle: Pair<Float, Float>,
        safetyFlag: Int,
        carList: List<Pair<Int, Int>>,
        correctedPoints: Pair<Pair<Int, Int>?, Pair<Int, Int>?>
    ): Triple<Bitmap, Int, Int>{
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        var flag = safetyFlag
        var cFlag = 0
        val unsafeZone = 0.35
        var unsafeCarZone = 0.35

        var leftCorrectedPoint = correctedPoints.first
        var rightCorrectedPoint = correctedPoints.second
        var leftCheckPoint = Pair(0, 0)
        var rightCheckPoint = Pair(0, 0)
        for (point in leftLane){
            if (carMiddle.second-5 < point.second && point.second < carMiddle.second+5){
                leftCheckPoint = point
                break
            }
        }
        for (point in rightLane){
            if (carMiddle.second-5 < point.second && point.second < carMiddle.second+5){
                rightCheckPoint = point
                break
            }
        }
        if (leftCorrectedPoint != null) {
            if (leftCorrectedPoint.second  > 0.7*bitmap.height){
                unsafeCarZone = 0.25
            }
        }

        val laneWidth = rightCheckPoint.first - leftCheckPoint.first
        if (leftCheckPoint.first + unsafeZone*laneWidth  <= carMiddle.first.toInt() && carMiddle.first.toInt() <= rightCheckPoint.first - unsafeZone*laneWidth){
            flag = 0
            drawText(canvas, bitmap.height, bitmap.width, "You're driving in the middle.", Color.BLACK)
        }else if (carMiddle.first.toInt() <= (leftCheckPoint.first + unsafeZone*laneWidth)){
            if (flag == 0 || flag == 1){
                flag = 1
                drawText(canvas, bitmap.height, bitmap.width, "Attention! You will change the lane to the left.", Color.BLACK)
            }
            else if (flag == 2){
                flag = 2
                drawText(canvas, bitmap.height, bitmap.width, "Attention! You will change the lane to the right.", Color.BLACK)
            }
        }else if (carMiddle.first.toInt() >= (rightCheckPoint.first - unsafeZone*laneWidth)){
            if (flag == 0 || flag == 2){
                flag = 2
                drawText(canvas, bitmap.height, bitmap.width, "Attention! You will change the lane to the right.", Color.BLACK)
            }
            else if (flag == 1) {
                flag = 1
                drawText(canvas, bitmap.height, bitmap.width, "Attention! You will change the lane to the left.", Color.BLACK)
            }
        }
        // car warning Detection
        for(point in carList){
            if (rightCorrectedPoint != null) {
                if (leftCorrectedPoint != null) {
                    if(point.first > leftCorrectedPoint.first && point.first < rightCorrectedPoint.first){
                        if(bitmap.height - point.second < unsafeCarZone*bitmap.height){
                            cFlag = 1
                            break
                        }
                    }
                }
            }
        }
        return Triple(mutableBitmap, flag, cFlag)
    }

    fun drawText(canvas: Canvas, height: Int, width: Int, text: String, textColor: Int){
        val paint = Paint().apply {
            color = textColor
            textSize = (height/20).toFloat()
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            isAntiAlias = true
        }
        var textWidth = paint.measureText(text)
        var textSiz = (height/20).toFloat()
        while (textWidth+20 > width){
            paint.textSize = textSiz--
            textWidth = paint.measureText(text)
        }
        val rectPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
            alpha = 128
        }

        val x = (width/2).toFloat() - textWidth/2
        val y = (height/10).toFloat()

        val rectLeft = x - 16
        val rectTop = y + paint.fontMetrics.top - 16
        val rectRight = x + textWidth + 16
        val rectBottom = y + paint.fontMetrics.bottom + 16

        canvas.drawRect(rectLeft, rectTop, rectRight, rectBottom, rectPaint)
        canvas.drawText(text, x, y, paint)
    }

    fun findHighestPixel(mask: Array<FloatArray>): Pair<Int, Int> {
        val width = mask.size
        val height = if (width > 0) mask[0].size else 0

        for (i in 0 until height) {
            var consecutiveOnes = 0
            for (j in 0 until width) {
                if (mask[j][i] == 1f) {
                    consecutiveOnes++
                    if (consecutiveOnes >= width*0.1) { //
                        return Pair(j - 9, i)
                    }
                } else {
                    consecutiveOnes = 0
                }
            }
        }
        return Pair(0, 0)
    }

    fun findHighestPixelClarified(mask: Array<FloatArray>, topLeftLine: Pair<Int, Int>, topRightLine: Pair<Int, Int>): Pair<Int, Int> {
        val width = mask.size
        val height = if (width > 0) mask[0].size else 0

        for (i in 0 until height) {
            var consecutiveOnes = 0
            for (j in 0 until width) {
                if (mask[j][i] == 1f) {
                    consecutiveOnes++
                    if (consecutiveOnes >= width*0.1 && topLeftLine.first < j && j < topRightLine.first) {
                        return Pair(j - (consecutiveOnes/2), i)
                    }
                } else {
                    consecutiveOnes = 0
                }
            }
        }
        return Pair(0, 0)
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

    fun averageLine(setOfLines: List<List<Pair<Int, Int>>>): MutableList<Pair<Int, Int>> {
        var b = 0
        var c = 0
        val workList = mutableListOf<List<Pair<Int, Int>>>()
        val finalList = mutableListOf<Pair<Int, Int>>()

        for (i in 0 until 3) {
            workList.add(setOfLines[setOfLines.size - i - 1])
        }

        for (i in 0 until 3) {
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

    fun approximateLine(setOfLines: List<List<Pair<Int, Int>>>, degree: Int, horizon1: Pair<Int, Int>, horizon2: Pair<Int, Int>, string: String): List<Pair<Int, Int>> {
        val iksy = mutableListOf<Int>()
        val igreki = mutableListOf<Int>()
        var polyLine = mutableListOf<Pair<Int, Int>>()
        var val1 = 0
        var val2 = 0

        if (setOfLines.size > 3) {
            val avgLine = averageLine(setOfLines)
            for (point in avgLine) {
                iksy.add(point.first)
                igreki.add(point.second)
            }
            val polynomial = fitPolynomial(igreki.map { it.toDouble() }.toDoubleArray(), iksy.map { it.toDouble() }.toDoubleArray(), degree)
            val1 = polynomial.value(horizon1.second.toDouble()).toInt()
            val2 = polynomial.value(horizon2.second.toDouble()).toInt()

            val ttt = (horizon1.second..horizon2.second).toList()
            for (x in ttt) {
                if (string == "left") {
                    if (polynomial.value(x.toDouble()).toInt() in val2..val1) {
                        polyLine.add(polynomial.value(x.toDouble()).toInt() to x)
                    }
                } else if (string == "right") {
                    if (polynomial.value(x.toDouble()).toInt() in val1..val2) {
                        polyLine.add(polynomial.value(x.toDouble()).toInt() to x)
                    }
                }
            }
        }

        return polyLine
    }

    fun fitPolynomial(igreki: DoubleArray, iksy: DoubleArray, degree: Int): PolynomialFunction {
        val obs = WeightedObservedPoints()
        for (i in igreki.indices) {
            obs.add(igreki[i], iksy[i])
        }
        val fitter = PolynomialCurveFitter.create(degree)
        return PolynomialFunction(fitter.fit(obs.toList()))
    }

    fun appendingListIfFoundOrNot(sidePointsList: MutableList<Pair<Int, Int>>, previousLinesList: MutableList<MutableList<Pair<Int, Int>>>) {
        if (sidePointsList.size > 6) {
            previousLinesList.add(sidePointsList.toMutableList())
        }
        if (sidePointsList.size <= 6 && previousLinesList.isNotEmpty()) {
            sidePointsList.clear()
            sidePointsList.addAll(previousLinesList.last())
            previousLinesList.add(sidePointsList.toMutableList())
        }
    }

    fun clearIfTooMany(listOfLists: MutableList<MutableList<Pair<Int, Int>>>, threshold: Int): MutableList<MutableList<Pair<Int, Int>>> {
        if (listOfLists.size > threshold) {
            listOfLists.removeAt(0)
        }
        return listOfLists
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

    fun processDetection(bitmap: Bitmap, pred: ArrayList<FloatArray>, confThreshold: Float,
                         ratiow: Float, ratioh: Float, classes: List<String>,
                         height_txt: Int): Pair<Bitmap, List<Pair<Int, Int>>> {
        val boxes = mutableListOf<Rect2d>()
        val confidences = mutableListOf<Float>()
        val classIds = mutableListOf<Int>()
        var processedBitmap = bitmap
        val carList = mutableListOf<Pair<Int, Int>>()
        var returnPair: Pair<Bitmap, List<Pair<Int, Int>>>

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
            val carPoint = Pair(left+(width/2), top+height)
            carList.add(carPoint)

            processedBitmap = drawPred(processedBitmap, classIds[i], confidences[i], left, top, left + width, top + height, classes, height_txt)
        }
        returnPair = Pair(processedBitmap, carList)
        return returnPair
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
                                ((transposedArray[a][b][yy][xx][0] * 2.0 - 0.5 + grid[j][yy][xx][0]) * stride[j]).toFloat()
                            transposedArray[a][b][yy][xx][1] =
                                ((transposedArray[a][b][yy][xx][1] * 2.0 - 0.5 + grid[j][yy][xx][1]) * stride[j]).toFloat()
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

    fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

}