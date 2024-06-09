package com.example.yolopv2app

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.YOLOPv2App.R
import com.example.YOLOPv2App.databinding.CameraActivityBinding
import kotlinx.coroutines.*
import org.opencv.android.OpenCVLoader
import java.lang.Runnable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

data class CameraResult(
    var outputBitmap: Bitmap,
    var setOfLanesLeft : MutableList<MutableList<Pair<Int, Int>>>,
    var setOfLanesRight : MutableList<MutableList<Pair<Int, Int>>>,
    var safetyFlag: Int,
    var carFlag: Int
)

class CameraActivity : AppCompatActivity() {
    private lateinit var binding: CameraActivityBinding

    private val backgroundExecutor: ExecutorService by lazy { Executors.newSingleThreadExecutor() }
    private val scope = CoroutineScope(Job() + Dispatchers.Main)

    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null

    private lateinit var classes:List<String>
    private lateinit var carWarningImage: Bitmap
    private lateinit var yolopv2Result: CameraResult

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = CameraActivityBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)

        yolopv2Result = CameraResult(
            outputBitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888),
            setOfLanesLeft = mutableListOf(),
            setOfLanesRight = mutableListOf(),
            safetyFlag = 0,
            carFlag = 0
        )
        classes = readClasses();
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)
        OpenCVLoader.initDebug()

        carWarningImage = BitmapFactory.decodeResource(resources, R.drawable.car_warning)
        carWarningImage = Bitmap.createScaledBitmap(carWarningImage, 120, 120, true)

        // Request Camera permission
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Get the display rotation and set size
            val display = (getSystemService(Context.WINDOW_SERVICE) as WindowManager).defaultDisplay
            val rotation = display.rotation
            var size = Size(900, 1700)
            if (rotation == 1 || rotation ==3){
                size = Size(1700, 900)
            }

            imageCapture = ImageCapture.Builder()
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(size)
                .build()
            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this, cameraSelector, imageCapture, imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

            setORTAnalyzer()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        ortEnv.close()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }

        }
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.yolopv2_192x320
        Log.d("YOLOPV2", "Yolopv2 192x320 is loaded!")
        return resources.openRawResource(modelID).readBytes()
    }

    private fun setORTAnalyzer(){
        scope.launch {
            imageAnalysis?.clearAnalyzer()
            imageAnalysis?.setAnalyzer(
                backgroundExecutor,
                CameraYolopv2Detector(::updateUI, ortEnv, ortSession, classes, yolopv2Result.setOfLanesLeft, yolopv2Result.setOfLanesRight, yolopv2Result.safetyFlag, yolopv2Result.carFlag)
            )
        }
    }
    private fun updateUI(bitmap: Pair<Bitmap, Int>) {
        runOnUiThread {
            val imageView = binding.imageView
            val canvas = Canvas(bitmap.first)
            val paint = Paint()
            paint.alpha = 80
            if(bitmap.second == 1){
                paint.alpha = 255
            }
            canvas.drawBitmap(carWarningImage, 10f, bitmap.first.height - 10f - carWarningImage.height, paint)
            imageView.setImageBitmap(bitmap.first)
        }
    }

    private fun readClasses(): List<String> {
        return resources.openRawResource(R.raw.classes).bufferedReader().readLines()
    }

    companion object {
        const val TAG = "YOLOPv2Detection"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
